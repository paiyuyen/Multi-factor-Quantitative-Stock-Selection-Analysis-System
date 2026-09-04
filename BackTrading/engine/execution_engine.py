"""回测引擎 — 撮合执行引擎

从 _run_single_backtest 嵌套函数提取的挂单队列管理与集合竞价撮合逻辑。

职责：
  - 挂单队列管理（买入/卖出 pending 列表）
  - 集合竞价撮合（一字板/触板/正常开盘）
  - 成交统计收集器（stats_sink 对接）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine.cost_calculator import CostAccum, CostCalculator
from BackTrading.engine.position_manager import PositionState
from BackTrading.limit_pricing import lot_size_for


@dataclass
class AuctionFillConfig:
    """集合竞价成交率配置。"""
    seal_sell_ratio: float = 0.05  # 一字涨停/跌停封板时卖出可成交量比例
    seal_buy_ratio: float = 0.02   # 一字涨停/跌停封板时买入可成交量比例
    tradable_ratio: float = 0.30   # 触板后炸板可成交量比例
    intraday_ratio: float = 0.10   # 盘中冲板可成交量比例
    seal_decay: float = 0.5        # 连续板衰减系数


@dataclass
class ExecutionEngine:
    """撮合执行引擎（消除 nonlocal 闭包 + 独立可测试）。

    构造时注入所有依赖，运行时无外部状态突变。
    """

    cost_calculator: CostCalculator
    position: PositionState
    fill_config: AuctionFillConfig = field(default_factory=AuctionFillConfig)

    # 挂单队列（通过 reset 重置）
    pending_sells: list[dict[str, Any]] = field(default_factory=list)
    pending_buys: list[dict[str, Any]] = field(default_factory=list)

    # ── stats_sink 统计收集 ──
    stats_sink: dict[str, Any] | None = None

    def reset(self) -> None:
        """回测开始时清空挂单队列和统计。"""
        self.pending_sells.clear()
        self.pending_buys.clear()
        self.cost_calculator.accumulator.reset()

    # ── sunk 辅助 ──────────────────────────────────

    def sink_inc(self, key: str, delta: int = 1) -> None:
        if self.stats_sink is None:
            return
        self.stats_sink[key] = self.stats_sink.get(key, 0) + delta

    def sink_val(self, key: str, value: float) -> None:
        if self.stats_sink is None:
            return
        self.stats_sink[key] = self.stats_sink.get(key, 0.0) + value

    def sink_worst(self, key: str, value: float) -> None:
        if self.stats_sink is None:
            return
        self.stats_sink[key] = max(self.stats_sink.get(key, 0.0), value)

    # ── AUCTION FILL 辅助 ───────────────────────

    def auction_fill_for(
        self,
        sym: str,
        open_px: float,
        limit_up: float,
        limit_down: float,
        side: str = "sell",
    ) -> float:
        """开盘触板日集合竞价可成交量比例。

        P1-2: 一字板场景拆分买卖方向成交率。
        """
        eps = 1e-9
        is_seal_up = abs(open_px - limit_up) < eps
        is_seal_down = abs(open_px - limit_down) < eps

        # 一字涨停
        if is_seal_up:
            if side == "sell":
                return self.fill_config.seal_sell_ratio
            else:
                return self.fill_config.seal_buy_ratio
        # 一字跌停
        if is_seal_down:
            if side == "sell":
                return self.fill_config.seal_sell_ratio
            else:
                return self.fill_config.seal_buy_ratio

        # 触板方向判定
        if open_px >= limit_up - eps:
            return self.fill_config.tradable_ratio if side == "buy" else self.fill_config.seal_sell_ratio
        if open_px <= limit_down + eps:
            return self.fill_config.tradable_ratio if side == "sell" else self.fill_config.seal_buy_ratio

        return 1.0

    def is_seal_up(self, open_px: float, limit_up: float) -> bool:
        return abs(open_px - limit_up) < 1e-9

    def is_seal_down(self, open_px: float, limit_down: float) -> bool:
        return abs(open_px - limit_down) < 1e-9

    # ── 成交价计算 ──────────────────────────────

    def exec_price_for(
        self,
        day_data: pd.DataFrame,
        j: int,
        close_raw: np.ndarray,
        exec_model: str,
    ) -> float:
        """根据执行模型计算成交价。"""
        import math

        if exec_model == "next_open":
            v = float("nan")
            if "open" in day_data.columns:
                o = float(day_data["open"].values[j])
                if np.isfinite(o) and o > 0:
                    v = o
            if np.isfinite(v) and v > 0:
                return round(v, 2)
            return v

        if exec_model == "vwap":
            c = float(close_raw[j])
            o = float(day_data["open"].values[j]) if "open" in day_data.columns else c
            h = float(day_data["high"].values[j]) if "high" in day_data.columns else c
            l = float(day_data["low"].values[j]) if "low" in day_data.columns else c

            vwap = None
            if "amount" in day_data.columns and "volume" in day_data.columns:
                amt = float(day_data["amount"].values[j])
                vol = float(day_data["volume"].values[j])
                if np.isfinite(amt) and np.isfinite(vol) and vol > 0 and amt > 0:
                    vwap = amt / vol
                    if not (l - 0.01 <= vwap <= h + 0.01):
                        vwap = None

            if vwap is not None:
                return round(vwap, 2)
            # 回退典型价
            typical = (o + h + l + c) / 4.0
            return round(typical, 2)

        return float("nan")

    # ── 核心撮合 ─────────────────────────────────

    def flush_pending(
        self,
        dt: str,
        day_data: pd.DataFrame,
        syms_str: np.ndarray,
        idx_arr: np.ndarray,
        close_adj: np.ndarray,
        close_raw: np.ndarray,
        open_arr: np.ndarray | None,
        volume_arr: np.ndarray,
        at_limit_up: np.ndarray,
        at_limit_down: np.ndarray,
        limit_up: np.ndarray,
        limit_down: np.ndarray,
        adj_ok: np.ndarray,
        has_volume: np.ndarray,
        amount_ma20: np.ndarray | None,
        vol_mult: np.ndarray | None,
        limit_tag: np.ndarray | None,
        exec_model: str,
        resume_gap_up_arr: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """次日开盘撮合：先卖后买。

        Returns:
            (buy_value, sell_value) 当日买卖金额。
        """
        if not self.pending_sells and not self.pending_buys:
            return 0.0, 0.0

        buy_val = sell_val = 0.0
        sym_row = {s: j for j, s in enumerate(syms_str)}

        # ── 卖出 ──
        remaining_sells: list[dict[str, Any]] = []
        for p in self.pending_sells:
            sym = p["sym"]
            if sym not in sym_row:
                remaining_sells.append(p)
                continue
            jj = sym_row[sym]
            si = p["si"]

            # 停牌/停牌复牌逻辑
            if not adj_ok[jj]:
                remaining_sells.append(p)
                continue

            # 计算成交价
            if open_arr is not None:
                px = float(open_arr[jj])
                if not np.isfinite(px) or px <= 0:
                    remaining_sells.append(p)
                    continue
            else:
                px = float(close_raw[jj])

            # 一字跌停 → 部分成交
            if self.is_seal_down(px, limit_down[jj]):
                ratio = self.auction_fill_for(sym, px, limit_up[jj], limit_down[jj], "sell")
                self.sink_inc("seal_sell_partial", 1)
                sell_shares = int((int(self.position.pos_shares[si]) * ratio))
                lot = lot_size_for(sym)
                sell_shares = max(0, sell_shares // lot * lot)
                if sell_shares <= 0:
                    sell_shares = int(self.position.pos_shares[si])
                tv = sell_shares * px

                adv = self.position.current_adv(sym)
                vol = float(volume_arr[jj])
                amount = float(amount_ma20[jj]) if amount_ma20 is not None else None
                amp = float(vol_mult[jj]) if vol_mult is not None else 1.0

                net, cost_t = self.cost_calculator.sell_proceeds_and_cost(
                    sym, tv, vol, adv, amount_ma20=amount, dt=str(dt),
                    volatility_multiplier=amp,
                )
                p["cash_ref"][0] += net
                sell_val += tv
                self.position.pos_shares[si] -= sell_shares
                p["qty_sold"] = sell_shares
                # 如果全卖完则移出队列
                if int(self.position.pos_shares[si]) <= 0:
                    continue
                else:
                    remaining_sells.append(p)
                    continue

            # 正常成交
            sh = int(self.position.pos_shares[si])
            if sh <= 0:
                continue

            tv = sh * px
            adv = self.position.current_adv(sym)
            vol = float(volume_arr[jj])
            amount = float(amount_ma20[jj]) if amount_ma20 is not None else None
            amp = float(vol_mult[jj]) if vol_mult is not None else 1.0

            net, cost_t = self.cost_calculator.sell_proceeds_and_cost(
                sym, tv, vol, adv, amount_ma20=amount, dt=str(dt),
                volatility_multiplier=amp,
            )
            p["cash_ref"][0] += net
            sell_val += tv
            lot = lot_size_for(sym)
            self.position.pos_shares[si] = 0

        self.pending_sells = remaining_sells

        # ── 买入 ──
        remaining_buys: list[dict[str, Any]] = []
        for p in self.pending_buys:
            sym = p["sym"]
            if sym not in sym_row:
                remaining_buys.append(p)
                continue
            jj = sym_row[sym]
            si = p.get("si", -1)

            # 复牌高开禁买
            if resume_gap_up_arr is not None and resume_gap_up_arr[jj]:
                remaining_buys.append(p)
                continue

            # 一字涨停 → 部分成交
            if open_arr is not None and self.is_seal_up(float(open_arr[jj]), limit_up[jj]):
                ratio = self.auction_fill_for(sym, float(open_arr[jj]), limit_up[jj], limit_down[jj], "buy")
                self.sink_inc("seal_buy_partial", 1)

                px = float(open_arr[jj])
                # 按比例计算可买入金额
                avail_tv = float(p["tv"]) * ratio
                shares = int(avail_tv / px) // lot * lot
                lot = lot_size_for(sym)
                if shares < lot:
                    remaining_buys.append(p)
                    continue

                cash_needed = shares * px
                cost = self.cost_calculator.buy_cost(
                    sym, cash_needed, shares, self.position.current_adv(sym),
                    amount_ma20=float(amount_ma20[jj]) if amount_ma20 is not None else None,
                    dt=str(dt),
                    volatility_multiplier=float(vol_mult[jj]) if vol_mult is not None else 1.0,
                )
                if p["cash_ref"][0] < cash_needed + cost:
                    remaining_buys.append(p)
                    continue

                p["cash_ref"][0] -= (cash_needed + cost)
                self.position.pos_shares[si] += shares
                buy_val += cash_needed
                p["qty_bought"] = shares
                if p.get("filled_buy"):
                    pass
                else:
                    # 部分成交后移出
                    pass
                continue

            # 正常买入
            px = self.exec_price_for(day_data, jj, close_raw, exec_model)
            if not np.isfinite(px) or px <= 0:
                remaining_buys.append(p)
                continue

            lot = lot_size_for(sym)
            shares = int(float(p["tv"]) / px) // lot * lot

            if shares < lot:
                # 现金不足 → 缩减
                # P1.1 简化处理
                remaining_buys.append(p)
                continue

            # ADV 约束
            adv_val = self.position.current_adv(sym)
            if adv_val > 100:
                adv_amount = adv_val * px
                max_order_pct = 0.10  # default
                max_shares_vol = int(adv_val * max_order_pct) // lot * lot
                shares = min(shares, max_shares_vol)
                if shares < lot:
                    remaining_buys.append(p)
                    continue

            tv = shares * px
            cost = self.cost_calculator.buy_cost(
                sym, tv, shares, self.position.current_adv(sym),
                amount_ma20=float(amount_ma20[jj]) if amount_ma20 is not None else None,
                dt=str(dt),
                volatility_multiplier=float(vol_mult[jj]) if vol_mult is not None else 1.0,
            )
            if p["cash_ref"][0] < tv + cost:
                # 现金不足 → 缩减
                max_affordable = p["cash_ref"][0] / (1.001)  # 估算
                shares = int(max_affordable / px) // lot * lot
                if shares < lot:
                    remaining_buys.append(p)
                    continue
                tv = shares * px
                cost = self.cost_calculator.buy_cost(
                    sym, tv, shares, self.position.current_adv(sym),
                    amount_ma20=float(amount_ma20[jj]) if amount_ma20 is not None else None,
                    dt=str(dt),
                    volatility_multiplier=float(vol_mult[jj]) if vol_mult is not None else 1.0,
                )
                if p["cash_ref"][0] < tv + cost:
                    remaining_buys.append(p)
                    continue

            p["cash_ref"][0] -= (tv + cost)
            self.position.pos_shares[si] = (shares // lot) * lot
            buy_val += tv
            self.position.buy_date[sym] = str(dt)

        self.pending_buys = remaining_buys

        return buy_val, sell_val