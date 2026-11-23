import sys
from types import SimpleNamespace

import pytest

from fastembed import TextEmbedding
from fastembed.common import onnx_model


def test_macos_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(onnx_model, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(onnx_model, "_performance_cores", lambda: 2)
    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    assert model.model.model.get_session_options().intra_op_num_threads == 2


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_performance_cores() -> None:
    assert onnx_model._performance_cores()
