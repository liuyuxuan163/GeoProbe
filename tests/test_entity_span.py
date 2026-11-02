"""测试：``entity_token`` 池化的 token 区间（2026-10-30 修的 bug）。

背景
----
``_entity_span`` 旧实现假设「位置 0 是 BOS」，返回 ``(1, full_length - 1)``。
但 Qwen2.5 的 tokenizer **不加 BOS**，于是它实际变成"丢掉第一个 subtoken、
再丢掉最后一个"：

    Tokyo (Tok+yo)          → 只取 `yo`
    Ho Chi Minh City (4 个) → 只取 `Chi`/`Minh`

319 词里只有 22 个是单 token，**51% 的地名 token 被丢弃**。

契约（改坏会在这里报错）
------------------------
1. 裸地名：span 必须覆盖**全部** token（``(0, n)``），一个都不能丢；
2. 有指令：span 必须**恰好**覆盖实体，不含指令前缀与尾随标点；
3. 兜底：拿不到 offsets 时不能抛异常，也不能返回空区间。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geoprobe.extract import _entity_span, _span_from_offsets, pool_activations  # noqa: E402


class FakeTokenizer:
    """只实现 offset_mapping 的最小 tokenizer 替身。"""

    def __init__(self, offsets):
        self._offsets = offsets

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert return_offsets_mapping, "实体定位必须要求 offsets"
        return {"offset_mapping": self._offsets}


def _offsets_from_pieces(pieces):
    """把 ["What", " are", ...] 变成 (字符起点, 字符终点) 列表。"""
    offsets, cursor = [], 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return offsets


# --------------------------------------------------------------------------- #
# 裸地名：不能丢任何 token
# --------------------------------------------------------------------------- #


class TestBareWordSpan:
    @pytest.mark.parametrize("n_tokens", [1, 2, 3, 4, 5, 8])
    def test_covers_all_tokens(self, n_tokens):
        span = _entity_span(None, "X", n_tokens, has_instruction=False)
        assert span == (0, n_tokens)

    def test_does_not_drop_first_or_last(self):
        """回归：旧实现返回 (1, n-1)，对 n>=3 会同时丢首尾。"""
        n = 4
        span = _entity_span(None, "Ho Chi Minh City", n, has_instruction=False)
        assert span != (1, n - 1), "不能退回到旧的 BOS 假设"
        assert span[0] == 0, "第一个 subtoken 必须被覆盖"
        assert span[1] == n, "最后一个 subtoken 必须被覆盖"

    def test_pooled_vector_is_mean_of_all_rows(self):
        hidden = np.array([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]], dtype=np.float32)
        span = _entity_span(None, "abc", len(hidden), has_instruction=False)
        pooled = pool_activations(hidden, "entity_token", span)
        assert np.allclose(pooled, hidden.mean(axis=0))


# --------------------------------------------------------------------------- #
# 有指令：必须恰好覆盖实体
# --------------------------------------------------------------------------- #


class TestInstructionSpan:
    TEXT = "What are the coordinates of Ho Chi Minh City?"
    PIECES = ["What", " are", " the", " coordinates", " of", " Ho", " Chi", " Minh", " City", "?"]

    def test_span_is_exactly_the_entity(self):
        entity = "Ho Chi Minh City"
        tokenizer = FakeTokenizer(_offsets_from_pieces(self.PIECES))
        span = _entity_span(
            tokenizer, self.TEXT, len(self.PIECES), True, entity=entity
        )
        assert span == (5, 9), "应覆盖 ' Ho',' Chi',' Minh',' City' 四个 token"

    def test_excludes_trailing_punctuation(self):
        entity = "Ho Chi Minh City"
        tokenizer = FakeTokenizer(_offsets_from_pieces(self.PIECES))
        start, end = _entity_span(
            tokenizer, self.TEXT, len(self.PIECES), True, entity=entity
        )
        assert end < len(self.PIECES), "问号不能被算进实体"

    def test_token_straddling_entity_start_is_included(self):
        """BPE 把「空格 + 实体首字母」并成一个 token 时，该 token 与实体有交集 → 必须纳入。"""
        text = "Coord: Paris"
        pieces = ["Coord", ":", " Paris"]  # 最后一个 token 同时含空格和实体
        tokenizer = FakeTokenizer(_offsets_from_pieces(pieces))
        span = _entity_span(tokenizer, text, len(pieces), True, entity="Paris")
        assert span == (2, 3)
        token_start, _ = _offsets_from_pieces(pieces)[2]
        assert token_start < text.find("Paris"), "这个 token 确实跨在实体起点之前"

    def test_missing_entity_falls_back_not_crash(self):
        tokenizer = FakeTokenizer(_offsets_from_pieces(self.PIECES))
        span = _entity_span(tokenizer, self.TEXT, len(self.PIECES), True, entity="Atlantis")
        assert span[0] == 0 and span[1] > 0

    def test_tokenizer_without_offsets_falls_back(self):
        class NoOffsets:
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                raise TypeError("unsupported")

        span = _entity_span(
            NoOffsets(), self.TEXT, len(self.PIECES), True, entity="Ho Chi Minh City"
        )
        assert span[0] == 0 and span[1] > 0


# --------------------------------------------------------------------------- #
# 纯函数 _span_from_offsets
# --------------------------------------------------------------------------- #


class TestSpanFromOffsets:
    OFFSETS = [(0, 4), (4, 8), (8, 12)]

    def test_exact_match(self):
        assert _span_from_offsets(self.OFFSETS, 4, 8) == (1, 2)

    def test_multi_token_span(self):
        assert _span_from_offsets(self.OFFSETS, 2, 10) == (0, 3)

    def test_partial_overlap_counts_as_hit(self):
        # 字符 6..7 与第二个 token (4,8) 有交集
        assert _span_from_offsets(self.OFFSETS, 6, 7) == (1, 2)

    def test_no_overlap_returns_none(self):
        assert _span_from_offsets(self.OFFSETS, 20, 30) is None

    def test_empty_offsets_returns_none(self):
        assert _span_from_offsets([], 0, 5) is None

    def test_zero_length_tokens_are_skipped(self):
        # 有些 tokenizer 会给出 (n, n) 的空 token
        spans = [(0, 4), (4, 4), (4, 8)]
        assert _span_from_offsets(spans, 5, 8) == (2, 3)
