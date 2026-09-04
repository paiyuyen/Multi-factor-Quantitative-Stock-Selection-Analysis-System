from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def compute_risk_metrics(
    equity_curve: list[dict[str, Any]],
    risk_free_rate: float = 0.03,
) -> dict[str, float]:
    """计算风险指标。

    #3 审计修复：新增 risk_free_rate 参数，Sharpe/Sortino 均扣除无风险利率。
    默认 3%（年化），可动态传入当期国债收益率。

    Args:
        equity_curve: 权益曲线 [{time, portfolio_value, turnover, ...}, ...]
        risk_free_rate: 年化无风险利率（默认 3%，中国 10 年期国债年化约 2.5-3.5%）
    """
    df = pd.DataFrame(equity_curve)
    if df.empty or "portfolio_value" not in df.columns or len(df) < 2:
        return {}

    vals = df["portfolio_value"].values.astype(float)
    # 非有限值（0 除、NaN、Inf）直接剔除对应点，避免静默污染全部指标
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return {}
    if not finite_mask.all():
        vals = vals[finite_mask]
    if vals[0] <= 0:
        return {}

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 2:
        return {}

    total_ret = vals[-1] / vals[0] - 1
    ann_factor = 244  # A股实际年化交易日数均值（非美股 252）
    mu = returns.mean() * ann_factor
    sigma = returns.std(ddof=1) * math.sqrt(ann_factor)

    # #3 修复：超额收益 Sharpe = (R_p - R_f) / σ
    excess_mu = mu - risk_free_rate
    sharpe = excess_mu / sigma if sigma > 0 else 0.0

    downside = returns[returns < 0]
    # P1 审计修复：无亏损日时 Sortino 截断为有限大值（100.0），避免优化器崩溃 / 数值不稳定
    # 原逻辑：float("inf") → DSR/PBO 污染 / scipy 优化器 NaN
    _SORTINO_CEILING = 100.0
    if len(downside) == 0:
        sortino = _SORTINO_CEILING if excess_mu > 0 else 0.0
    else:
        downside_std = downside.std(ddof=1) * math.sqrt(ann_factor)
        raw_sortino = excess_mu / downside_std if downside_std > 0 else (float("inf") if excess_mu > 0 else 0.0)
        sortino = min(raw_sortino, _SORTINO_CEILING)

    peak = np.maximum.accumulate(vals)  # type: ignore[arg-type]
    dd = (vals - peak) / peak
    max_dd = float(dd.min())

    # 数据异常告警：回撤 < -100% 意味着期中出现负净资产（正常价格数据不可能），
    # 通常由负复权价/负市值导致，提示检查上游数据而非真实亏损
    if max_dd < -1.0:
        logger.warning(f"净资产曲线 max_drawdown={max_dd:.4f} < -100%，存在负净资产点，"
                       f"疑似复权价数据异常（close_adj<=0），请检查数据源")

    peak_idx = int(np.argmax(peak))
    trough_idx = int(np.argmin(vals[peak_idx:])) + peak_idx if peak_idx < len(vals) - 1 else peak_idx
    dd_duration = int(trough_idx - peak_idx) if trough_idx > peak_idx else 0

    sorted_ret = np.sort(returns)
    var_95 = float(np.percentile(sorted_ret, 5))
    cvar_95 = float(sorted_ret[sorted_ret <= var_95].mean()) if np.any(sorted_ret <= var_95) else var_95

    # 几何年化（CAGR）与年化 Calmar：跨期可比，避免算术年化高估
    years = n / ann_factor
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else total_ret
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # ── 换手率 ──
    turnover = df["turnover"].values.astype(float) if "turnover" in df.columns else np.array([0.0])
    turnover = turnover[np.isfinite(turnover)]
    avg_turnover = float(turnover.mean()) if len(turnover) else 0.0
    max_turnover = float(turnover.max()) if len(turnover) else 0.0

    return {
        "total_return": round(total_ret, 6),
        "annual_return": round(cagr, 6),
        "annual_vol": round(sigma, 6),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4) if math.isfinite(sortino) else None,
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_duration": dd_duration,
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "avg_turnover": round(avg_turnover, 6),
        "max_turnover": round(max_turnover, 6),
        # #3 修复：输出 risk_free_rate 用于审计可追溯
        "risk_free_rate": risk_free_rate,
    }


def compute_trade_metrics(trade_log: list[dict[str, Any]]) -> dict[str, Any]:
    buys = [t for t in trade_log if t.get("action") == "buy"]
    sells = [t for t in trade_log if t.get("action", "").startswith("sell")]

    if not buys or not sells:
        return {"total_trades": 0}

    total = len(buys) + len(sells)

    # FIFO 按 symbol 配对：买入队列按成交顺序，卖出（含 sell_partial）按股数消耗买入，
    # 成本/金额按比例分摊，避免半仓卖出与全仓买入错配导致的 PnL 失真。
    from collections import defaultdict
    buy_queue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for b in buys:
        buy_queue.setdefault(b["symbol"], []).append(b)

    pnl = []
    for s in sells:
        sym = s["symbol"]
        shares = float(s.get("shares", 0))
        proceeds = float(s.get("value", 0))
        q = buy_queue.get(sym)
        if not q:
            continue
        if shares <= 0:
            # 旧日志无 shares 字段：退化为整笔配对
            b = q.pop(0)
            pnl.append(proceeds - b.get("value", 0) - b.get("cost", 0))
            continue
        remaining = shares
        while remaining > 1e-9 and q:
            b = q[0]
            b_sh = float(b.get("shares", shares))
            take = min(remaining, b_sh)
            frac = take / b_sh if b_sh > 0 else 1.0
            pnl.append(proceeds * (take / shares) - b.get("value", 0) * frac - b.get("cost", 0) * frac)
            remaining -= take
            if b_sh - take <= 1e-9:
                q.pop(0)
            else:
                b["shares"] = b_sh - take

    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]

    win_rate = len(wins) / len(pnl) if pnl else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1e-10
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float("inf")

    return {
        "total_trades": total,
        "buy_trades": len(buys),
        "sell_trades": len(sells),
        "win_rate": round(win_rate, 4),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4),
        "total_pnl": round(float(sum(pnl)), 2),
    }
