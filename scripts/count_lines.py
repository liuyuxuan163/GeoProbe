"""统计项目规模。"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def docstring_lines(path: Path) -> set[int]:
    lines: set[int] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    start = body[0].lineno
                    end = getattr(body[0], "end_lineno", start) or start
                    lines.update(range(start, end + 1))
    return lines


def measure(paths: list[Path]) -> tuple[int, int, int]:
    files = [p for p in paths if p.is_file()]
    total = 0
    logic = 0
    for path in files:
        source = path.read_text(encoding="utf-8", errors="ignore")
        rows = source.splitlines()
        total += len(rows)
        skip = docstring_lines(path)
        logic += sum(
            1
            for index, raw in enumerate(rows, 1)
            if raw.strip() and not raw.strip().startswith("#") and index not in skip
        )
    return len(files), total, logic


def main() -> int:
    groups = [
        ("源码 geoprobe/", list((ROOT / "geoprobe").rglob("*.py"))),
        ("测试 tests/", list((ROOT / "tests").rglob("*.py"))),
        ("脚本 scripts/", list((ROOT / "scripts").rglob("*.py"))),
    ]
    print(f"{'目录':<20}{'文件':>5}{'总行':>7}{'代码行':>8}")
    print("-" * 40)
    for name, paths in groups:
        count, total, logic = measure(paths)
        print(f"{name:<20}{count:>5}{total:>7}{logic:>8}")

    _, data_total, _ = measure(list((ROOT / "data").glob("*.json")))
    print(f"{'数据 data/':<20}{len(list((ROOT / 'data').glob('*.json'))):>5}{data_total:>7}{'—':>8}")

    print("\n各模块：")
    for path in sorted((ROOT / "geoprobe").glob("*.py")):
        _, total, logic = measure([path])
        print(f"  {path.name:<20}{total:>6} 行（代码 {logic}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
