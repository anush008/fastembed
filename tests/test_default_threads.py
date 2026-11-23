import sys

import pytest

from fastembed.common import onnx_model
from fastembed.common.onnx_model import _default_cpu_threads


def test_default_threads_other_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert _default_cpu_threads() is None


@pytest.mark.parametrize(
    "sysctl, expected",
    [
        ({"hw.nperflevels": 2, "hw.perflevel0.physicalcpu": 8}, 8),
        ({"hw.nperflevels": 1, "hw.perflevel0.physicalcpu": 3}, None),
        ({}, None),
    ],
)
def test_default_threads_macos(
    monkeypatch: pytest.MonkeyPatch, sysctl: dict[str, int], expected: int | None
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(onnx_model, "_macos_sysctl_int", sysctl.get)
    assert _default_cpu_threads() == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="reads macOS sysctl values")
def test_macos_sysctl_reads_cpu_count() -> None:
    assert (onnx_model._macos_sysctl_int("hw.physicalcpu") or 0) > 0
