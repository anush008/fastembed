"""Count the threads that an onnxruntime session starts, with and without `threads`."""

import os
import subprocess
import sys
import time

from fastembed import TextEmbedding
from fastembed.common import onnx_model


def thread_count() -> int:
    if sys.platform == "darwin":
        out = subprocess.run(["ps", "-M", str(os.getpid())], capture_output=True, text=True)
        return len(out.stdout.splitlines()) - 1
    return len(os.listdir(f"/proc/{os.getpid()}/task"))


mode = sys.argv[1]
if mode == "ort-default":
    # keep the onnxruntime default thread count
    onnx_model._default_cpu_threads = lambda: None

before = thread_count()
kwargs = {"threads": int(mode)} if mode.isdigit() else {}
model = TextEmbedding("BAAI/bge-small-en-v1.5", **kwargs)
docs = ["vector search engine stores embeddings " * 20] * 256
list(model.embed(docs[:32]))
t0 = time.perf_counter()
list(model.embed(docs, batch_size=64))
elapsed = time.perf_counter() - t0
print(
    f"mode={mode:<12} threads_before={before} threads_after={thread_count()} "
    f"docs_per_s={len(docs) / elapsed:.1f}"
)
