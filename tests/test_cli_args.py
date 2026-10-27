"""测试：命令行参数完整性。

**这一组测试针对一个真实的崩溃。**

现象::

    python -m geoprobe layer-sweep --model-name D:\\models\\Qwen2.5-3B-Instruct
    [FAIL] AttributeError: 'Namespace' object has no attribute 'test_size'

原因：`cmd_layer_sweep` 里写了 ``args.test_size``，但 ``--test-size`` 只注册在
``run`` 子命令上，``layer-sweep`` 没有。于是 **提取与缓存都成功之后**才崩 ——
现象特别误导：`activations.qwen2.5-3b-instruct.*.npz` 是好的，
但 `layer_sweep.qwen2.5-3b-instruct.*.json` 永远不生成。

这类 bug 的共性是"命令读取了一个自己没声明的参数"，所以这里用
**解析默认 argv 后检查属性是否存在** 来直接钉住它。
"""

from __future__ import annotations

import pytest

from geoprobe.cli import build_parser


def parse_defaults(subcommand: str, extra: list[str] | None = None):
    """用最少的参数解析某个子命令，拿到 Namespace。"""
    parser = build_parser()
    argv = [subcommand] + (extra or [])
    # run / layer-sweep 需要真实模型；用 --mock 避免触碰网络与磁盘
    if subcommand in ("run", "layer-sweep"):
        argv.append("--mock")
    return parser.parse_args(argv)


class TestArgumentCompleteness:
    """每个命令都必须声明它用到的参数。"""

    def test_layer_sweep_has_test_size(self):
        """**核心回归**：layer-sweep 必须声明 --test-size。

        历史 bug：`cmd_layer_sweep` 在构造 _meta 时读 `args.test_size`，
        但 `--test-size` 只在 `run` 上注册 → AttributeError。
        """
        args = parse_defaults("layer-sweep")
        assert hasattr(args, "test_size"), "layer-sweep 缺少 --test-size，_meta 构造会崩"
        assert args.test_size == pytest.approx(0.2)

    def test_layer_sweep_has_step(self):
        args = parse_defaults("layer-sweep")
        assert hasattr(args, "step")
        assert args.step == 1

    def test_run_has_its_own_flags(self):
        args = parse_defaults("run")
        for name in ("test_size", "from_cache", "with_mlp", "title", "out"):
            assert hasattr(args, name), f"run 缺少 --{name.replace('_', '-')}"

    def test_shared_flags_on_both_commands(self):
        """run 与 layer-sweep 共享的参数必须两边都有。"""
        shared = (
            "mock",
            "pooling",
            "target_encoding",
            "layer",
            "device",
            "dtype",
            "model_name",
            "repeats",
            "test_size",
            "json",
        )
        for subcommand in ("run", "layer-sweep"):
            args = parse_defaults(subcommand)
            for name in shared:
                assert hasattr(args, name), f"{subcommand} 缺少 --{name.replace('_', '-')}"

    def test_logit_lens_has_its_flags(self):
        parser = build_parser()
        args = parser.parse_args(["logit-lens"])
        for name in ("mode", "limit", "dtype", "device", "model_name"):
            assert hasattr(args, name), f"logit-lens 缺少 --{name.replace('_', '-')}"
        assert args.mode == "bare"

    def test_models_and_check_parse(self):
        parser = build_parser()
        assert parser.parse_args(["models"]).func is not None
        assert parser.parse_args(["check"]).func is not None
        assert parser.parse_args(["info"]).func is not None


class TestLayerSweepSmoke:
    """layer-sweep 在 mock 下必须能完整跑完（含 _meta 构造）。"""

    def test_mock_layer_sweep_exit_ok(self, capsys):
        from geoprobe.cli import main

        code = main(["layer-sweep", "--mock", "--repeats", "1"])
        assert code == 0, "layer-sweep --mock 应正常返回"

    def test_run_mock_exit_ok(self):
        from geoprobe.cli import main

        code = main(["run", "--mock", "--repeats", "1"])
        assert code == 0, "run --mock 应正常返回"
