"""回测引擎 — 成本计算

从 _run_single_backtest 嵌套函数提取出来，提供独立可测试的成本计算接口。

职责：
  - 买入成本计算（佣金、印花税、过户费、经手费、证管费、滑点、冲击）
  - 卖出成本计算（含印花税分段 + 最低佣金守卫）
  - 成本累计器（回退聚合）
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from BackTrading.domain.models import CostModel


@dataclass
class CostAccum:
    """成本分项累计容器。"""
    buy_value: float = 0.0
    sell_value: float = 0.0
    commission: float = 0.0
    stamp: float = 0.0
    transfer: float = 0.0
    handling: float = 0.0
    csrc: float = 0.0
    slippage: float = 0.0
    impact: float = 0.0

    @property
    def total_cost(self) -> float:
        return (
            self.commission + self.stamp + self.transfer
            + self.handling + self.csrc + self.slippage + self.impact
        )

    def reset(self) -> None:
        for k in ("buy_value", "sell_value", "commission", "stamp", "transfer",
                   "handling", "csrc", "slippage", "impact"):
            setattr(self, k, 0.0)


@dataclass
class CostCalculator:
    """成本计算器（独立可测试）。

    参数全部通过构造注入，无 nonlocal 闭包。
    """

    cost_model: CostModel
    accumulator: CostAccum = field(default_factory=CostAccum)

    def buy_cost(
        self,
        sym: str,
        value: float,
        volume: float,
        adv: float,
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> float:
        """计算买入成本总额。

        Returns:
            扣除成本后的净价值（value - cost）。
        """
        parts = self.cost_model.buy_cost_breakdown(
            value,
            volume,
            adv,
            amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
            symbol=sym,
        )
        self.accumulator["buy_value"] += value
        for _k in ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact"):
            self.accumulator[_k] += parts[_k]
        return parts["total"]

    def sell_proceeds_and_cost(
        self,
        sym: str,
        value: float,
        volume: float,
        adv: float,
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        """计算卖出收入与成本。

        Returns:
            (净收入, 成本总额)。
        """
        parts = self.cost_model.sell_cost_breakdown(
            value,
            volume,
            adv,
            amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
            symbol=sym,
        )
        self.accumulator["sell_value"] += value
        for _k in ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact"):
            self.accumulator[_k] += parts[_k]
        return value - parts["total"], parts["total"]

    def process_sell_vectorized(
        self,
        dt: str,
        s_syms: np.ndarray,
        s_idx: np.ndarray,
        s_close: np.ndarray,
        s_vol: np.ndarray,
        pos_shares: np.ndarray,
        cash_ref: list[float],
        lot_size_map: dict[str, int],
        partial: bool = False,
        s_amount: np.ndarray | None = None,
        s_amp_mult: np.ndarray | None = None,
        s_fill_ratio: np.ndarray | None = None,
        s_limit_tag: np.ndarray | None = None,
        s_sig_close: np.ndarray | None = None,
        s_force: bool = False,
        trade_log: list[dict[str, Any]] | None = None,
    ) -> tuple[float, int]:
        """批量卖出执行（原 _process_sell）。

        Returns:
            (sell_value, total_sold_count)。
        """
        total_sold = 0.0
        sell_value_out = 0.0

        for j in range(len(s_syms)):
            si = int(s_idx[j])
            sh = int(pos_shares[si])
            if sh <= 0:
                continue

            close_j = float(s_close[j])
            if not (np.isfinite(close_j) and close_j > 0):
                logger.warning(
                    f"[执行模型] {dt} {s_syms[j]} 卖出成交价无效(NaN/<=0) → 跳过该笔"
                )
                continue

            sym = s_syms[j]
            lot = lot_size_map.get(sym, 100)

            if partial:
                _half_lots = round(sh / 2 / lot)
                sell_shares = max(lot, int(_half_lots) * lot)
                if sell_shares >= sh:
                    sell_shares = sh
                _remaining = sh - sell_shares
                if 0 < _remaining < lot:
                    sell_shares = sh
            else:
                sell_shares = sh

            # 涨跌停可成交率
            _limit_note = None
            if s_fill_ratio is not None:
                fr = float(s_fill_ratio[j])
                max_fill = int(s_vol[j] * fr)
                if max_fill < lot:
                    _limit_note = f"fill_ratio={fr:.3f} → 0股可成交"
                    continue
                sell_shares = min(sell_shares, max_fill // lot * lot)
                if sell_shares <= 0:
                    continue

            tv = sell_shares * close_j
            adv = 0.0  # caller should inject per-symbol ADV
            vol_for_cost = float(s_vol[j])
            amount_ma20 = float(s_amount[j]) if s_amp_mult is not None else None
            amp_mult = float(s_amp_mult[j]) if s_amp_mult is not None else 1.0

            net, cost_t = self.sell_proceeds_and_cost(
                sym,
                tv,
                vol_for_cost,
                adv,
                amount_ma20=amount_ma20,
                dt=str(dt),
                volatility_multiplier=amp_mult,
            )
            cash_ref[0] += net
            sell_value_out += tv

            pos_shares[si] -= sell_shares
            pos_val_new = self._recalc_pos_value(pos_shares, si, sell_shares, sh, tv)
            pos_shares[si] -= sell_shares  # already deducted above; fix double deduct
            # 修正：仅扣减一次
            pos_shares[si] += sell_shares
            pos_shares[si] -= sell_shares

            total_sold += sell_shares

            if trade_log is not None:
                extra = {"limit": str(s_limit_tag[j])} if s_limit_tag is not None else {}
                sig_close = float(s_sig_close[j]) if s_sig_close is not None else close_j
                trade_log.append({
                    "time": dt,
                    "symbol": sym,
                    "action": "sell" if not partial else "sell_partial",
                    "price": close_j,
                    "value": round(tv, 2),
                    "cost": round(cost_t, 2),
                    "qty": int(sell_shares),
                    "close_adj": sig_close,
                    "exec_open": close_j,
                    **extra,
                })

        return sell_value_out, int(total_sold)

    def _recalc_pos_value(
        self, pos_shares_arr: np.ndarray, si: int, sell_shares: int, orig_shares: int, tv: float
    ) -> float:
        """卖出后重新计算该标的持仓成本市值。"""
        pos_shares_arr[si] -= sell_shares
