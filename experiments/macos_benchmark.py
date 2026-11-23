"""Compare FastEmbed throughput across onnxruntime settings.

Run on a Mac:
    uv run python experiments/macos_benchmark.py

Each configuration runs in its own subprocess so that thread pools and
allocations of one configuration do not affect the next one.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

WORDS = (
    "vector search engine stores embeddings and returns the nearest neighbours of a query "
    "with filtering payload indexing quantization and distributed deployment options"
).split()


def make_documents(n: int, seed: int = 0) -> list[str]:
    import random

    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        length = rng.choice([8, 16, 32, 64, 128, 256])
        docs.append(" ".join(rng.choice(WORDS) for _ in range(length)))
    return docs


def perf_cores() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True
        )
        return int(out.stdout.strip())
    except (ValueError, OSError):
        return None


def patch_session_options(config: dict[str, str]) -> None:
    import onnxruntime as ort

    original = ort.SessionOptions

    def factory() -> "ort.SessionOptions":
        so = original()
        for key, value in config.items():
            so.add_session_config_entry(key, value)
        return so

    ort.SessionOptions = factory  # type: ignore[misc]


def run_one(cfg: dict, args: argparse.Namespace) -> dict:
    if cfg.get("session_config"):
        patch_session_options(cfg["session_config"])

    from fastembed import TextEmbedding

    kwargs = {}
    if cfg.get("threads") is not None:
        kwargs["threads"] = cfg["threads"]
    if cfg.get("providers") is not None:
        kwargs["providers"] = cfg["providers"]

    t0 = time.perf_counter()
    model = TextEmbedding(args.model, **kwargs)
    load_s = time.perf_counter() - t0

    docs = make_documents(args.docs)
    queries = make_documents(args.queries, seed=1)

    list(model.embed(docs[:64], batch_size=args.batch_size))  # warm up

    doc_runs = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        list(model.embed(docs, batch_size=args.batch_size, parallel=cfg.get("parallel")))
        doc_runs.append(time.perf_counter() - t0)

    query_runs = []
    for q in queries:
        t0 = time.perf_counter()
        list(model.query_embed(q))
        query_runs.append(time.perf_counter() - t0)

    best = min(doc_runs)
    return {
        "name": cfg["name"],
        "load_s": round(load_s, 3),
        "docs_per_s": round(args.docs / best, 1),
        "docs_s_runs": [round(x, 3) for x in doc_runs],
        "query_p50_ms": round(statistics.median(query_runs) * 1000, 2),
    }


def configs(args: argparse.Namespace) -> list[dict]:
    import onnxruntime as ort

    p = perf_cores()
    no_spin = {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }
    out = [{"name": "baseline"}]
    out.append({"name": "no_spin", "session_config": no_spin})
    out.append(
        {"name": "dynamic_block_base=4", "session_config": {"session.dynamic_block_base": "4"}}
    )
    cpus = os.cpu_count() or 1
    for t in sorted({1, 2, cpus // 2, cpus} - {0}):
        out.append({"name": f"threads={t}", "threads": t})
    if p:
        out.append({"name": f"threads={p} (P-cores)", "threads": p})
        out.append({"name": f"threads={p} + no_spin", "threads": p, "session_config": no_spin})
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        for units in ("ALL", "CPUAndGPU", "CPUAndNeuralEngine"):
            out.append(
                {
                    "name": f"coreml {units}",
                    "providers": [
                        (
                            "CoreMLExecutionProvider",
                            {"ModelFormat": "MLProgram", "MLComputeUnits": units},
                        ),
                        "CPUExecutionProvider",
                    ],
                }
            )
    if args.only:
        out = [c for c in out if any(s in c["name"] for s in args.only)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900, help="seconds per configuration")
    parser.add_argument("--only", nargs="*", help="run configurations whose name has this text")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        print("RESULT " + json.dumps(run_one(json.loads(args.child), args)), flush=True)
        return

    import onnxruntime as ort

    print(
        f"platform={platform.platform()} machine={platform.machine()} "
        f"cpu_count={os.cpu_count()} perf_cores={perf_cores()} ort={ort.__version__}"
    )
    print(f"providers={ort.get_available_providers()}")
    results = []
    for cfg in configs(args):
        cmd = [sys.executable, __file__, "--child", json.dumps(cfg)] + sys.argv[1:]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"{cfg['name']}: TIMEOUT after {args.timeout} s")
            continue
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"{cfg['name']}: FAILED\n{proc.stderr[-2000:]}")
            continue
        res = json.loads(line[len("RESULT ") :])
        results.append(res)
        print(json.dumps(res), flush=True)

    if results:
        base = results[0]
        print(f"\n{'config':<32}{'docs/s':>10}{'vs base':>10}{'query p50 ms':>14}{'load s':>9}")
        for r in results:
            ratio = r["docs_per_s"] / base["docs_per_s"]
            print(
                f"{r['name']:<32}{r['docs_per_s']:>10}{ratio:>9.2f}x"
                f"{r['query_p50_ms']:>14}{r['load_s']:>9}"
            )


if __name__ == "__main__":
    main()
