"""pytest 夹具：把项目根加入 sys.path，并提供不受系统临时目录限制的工作目录。

为什么不用 pytest 内置的 `tmp_path`
-----------------------------------
在受限的 Windows 环境下，系统临时目录会报
``PermissionError: [WinError 5]`` —— pytest 会去遍历它自己的 base temp 目录。
所以这里用"项目内 mkdir + 唯一名字"，绕开那套机制。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNTIME = ROOT / ".runtime" / "tests"
RUNTIME.mkdir(parents=True, exist_ok=True)


@pytest.fixture()
def workdir() -> Path:
    """可写的工作目录（凡是要落盘的测试都用它）。"""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = RUNTIME / f"case-{os.getpid()}-{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("无法创建测试工作目录")


@pytest.fixture(autouse=True)
def _isolate_temp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 temp 相关环境变量指向项目内目录。"""
    monkeypatch.setenv("TMPDIR", str(RUNTIME))
    monkeypatch.setenv("TEMP", str(RUNTIME))
    monkeypatch.setenv("TMP", str(RUNTIME))
