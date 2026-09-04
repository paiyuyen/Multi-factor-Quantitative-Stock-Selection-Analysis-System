"""回测引擎 — 仓位状态管理

职责：持仓数据、ADV 滚动均值、停牌盯市、市值计算。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.limit_pricing import lot_size_for


# ADV 滚动窗口（不含当日，日后再入账）
_ADV_WINDOW = 20


@dataclass
class PositionState:
    """所有持仓相关可变状态（非 local 闭包替代）。"""

    # ── 仓位核心 ──
    pos_shares: np.ndarray  # (n_syms,) 持仓股数
    pos_value: np.ndarray   # (n_syms,) 持仓成本市值
    symbols: np.ndarray     # (n_syms,) 股票代码
    sym_to_idx: dict[str, int]  # symbol → index

    # ── ADV 滚动均值 ──
    adv_state: dict[str, tuple[Any, float]] = field(default_factory=dict)

    # ── 停牌盯市 / K 线历史 ──
    last_close: dict[str, float] = field(default_factory=dict)
    prev_bar: dict[str, tuple[float, float]] = field(default_factory=dict)  # (raw_close, atr)
    prev_bar_adj: dict[str, tuple[float, float]] = field(default_factory=dict)  # (adj_close, atr)
    prev_bar_date: dict[str, str] = field(default_factory=dict)
    prev_af: dict[str, float] | None = None

    # ── 除权调整 ──
    pos_adjf: dict[str, float] = field(default_factory=dict)
    prev_af_guard: list[bool] = field(default_factory=lambda: [True])

    # ── 交易日期跟踪 ──
    buy_date: dict[str, str] = field(default_factory=dict)
    entry_buy_score: dict[str, float] = field(default_factory=dict)

    # ── 停牌天数 ──
    susp_days: dict[str, int] = field(default_factory=dict)

    # ── 止损线 ──
    prev_stop: dict[str, float] = field(default_factory=dict)

    @property
    def held_mask(self) -> np.ndarray:
        """持仓 > 0 的布尔掩码。"""
        return self.pos_shares > 0

    # ── ADV ──────────────────────────────────────────────

    def update_adv(self, sym: str, vol: float) -> None:
        """滚动 ADV 窗口：当日 bar 结束后入账，供次日使用。

        P2.7 修复：vol <= 0 时不推进滑动窗口（停牌/零量日 forward fill 上次有效 ADV），
        避免停牌期 0 值拉低 ADV → 复牌后冲击成本/分档上限失真。
        """
        dq, run = self.adv_state.get(sym, (None, 0.0))
        # P2.7：停牌/零量日不推进窗口，保持历史 ADV 不变
        if vol <= 0:
            return
        if dq is None:
            dq = deque(maxlen=_ADV_WINDOW)
        if len(dq) == dq.maxlen:
            run -= dq[0]
        dq.append(vol)
        run += vol
        self.adv_state[sym] = (dq, run)

    def current_adv(self, sym: str) -> float:
        """当前可用 ADV（前一日及之前窗口滚动均值）。"""
        dq, run = self.adv_state.get(sym, (None, 0.0))
        return (run / len(dq)) if dq else 0.0

    # ── 市值计算 ─────────────────────────────────────

    def calc_market_value(self, close_lookup: dict[str, float]) -> float:
        """按当前收盘价计算持仓总市值。"""
        mtm = 0.0
        for si in np.where(self.held_mask)[0]:
            s = self.symbols[si]
            px = close_lookup.get(s)
            if px is None or not np.isfinite(px) or px <= 0:
                px = self.last_close.get(s)
                if px is None or px <= 0:
                    continue
            mtm += px * self.pos_shares[si]
        return mtm

    def susp_position_value(self) -> float:
        """停牌期持仓市值（当日无行情标的）。"""
        val = 0.0
        for si in np.where(self.held_mask)[0]:
            s = self.symbols[si]
            if s not in self.last_close:
                continue
            val += self.last_close[s] * self.pos_shares[si]
        return val

    # ── 碎股检测与清理建议 ────────────────────────

    def fractional_positions(
        self, lot_size_map: dict[str, int] | None = None,
    ) -> list[tuple[str, int, int]]:
        """返回碎股持仓列表 [(symbol, shares, lot), ...]。"""
        result: list[tuple[str, int, int]] = []
        for si in np.where(self.held_mask)[0]:
            shares = int(self.pos_shares[si])
            if shares <= 0:
                continue
            s = self.symbols[si]
            lot = lot_size_map.get(s, lot_size_for(s)) if lot_size_map else lot_size_for(s)
            if 0 < shares < lot:
                result.append((s, shares, lot))
        return result
