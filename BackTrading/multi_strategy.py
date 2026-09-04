from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.prepare import _build_params, prepare_backtest_data

# 3 个子策略的因子权重配置（设计为低相关）
STRATEGY_WEIGHTS = {
    "momentum": {  # 动量策略
        "macd": 0.35, "momentum": 0.35, "moneyflow": 0.20,
        "quality": 0.05, "valuation": 0.05, "top_trader": 0.0,
    },
    "fundamental_reversal": {  # 基本面反转
        "macd": 0.05, "momentum": 0.05, "moneyflow": 0.10,
        "quality": 0.35, "valuation": 0.35, "top_trader": 0.0,
    },
    "defensive": {  # 低波防御
        "macd": 0.20, "momentum": 0.05, "moneyflow": 0.25,
        "quality": 0.25, "valuation": 0.10, "top_trader": 0.05,
    },
}


def run_multi_strategy_backtest(
    kline_df: pd.DataFrame,
    engine_cfg: EngineConfig,
    params: dict[str, float],
    trade_log: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
) -> dict[str, Any]:
    """多策略组合回测：运行 3 个子策略 + 上层资本分配。

    Args:
        kline_df: 全量 K 线数据
        engine_cfg: 引擎配置
        params: WFO 最佳参数
        trade_log: 输出 — 合并后的逐笔交易记录
        equity_curve: 输出 — 合并后的每日净值

    Returns:
        {strategy_name: sub_result} 各子策略详情
    """
    # 准备信号数据
    structured = _build_params(type("cfg", (), {})(), params)
    prepared = prepare_backtest_data(kline_df, params=structured, compute_exit_strategy=True, vectorized=True)

    # 止损价
    stop_mult = params.get("atr_stop_mult", 2.0)
    if "ATR" in prepared.columns:
        # P0-1：止损价与引擎比较基准统一到后复权空间（指标 ATR 亦为后复权）
        _stop_close = prepared["close_normal"] if "close_normal" in prepared.columns else prepared["close"]
        prepared["止损价"] = _stop_close - prepared["ATR"] * stop_mult
    else:
        prepared["止损价"] = 0.0

    sub_results = {}

    for strat_name, factor_weights in STRATEGY_WEIGHTS.items():
        # 用子策略权重重算综合评分
        strat_data = prepared.copy()
        _rerank_scores(strat_data, factor_weights)

        tl: list[dict[str, Any]] = []
        ec: list[dict[str, Any]] = []
        _run_single_backtest(strat_data, params, engine_cfg, tl, ec)

        from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics

        risk = compute_risk_metrics(ec) or {}
        trade = compute_trade_metrics(tl) or {}

        sub_results[strat_name] = {
            "sharpe": risk.get("sharpe_ratio", 0),
            "total_return": risk.get("total_return", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "win_rate": trade.get("win_rate", 0),
            "num_trades": trade.get("total_trades", 0),
            "trade_log": tl,
            "equity_curve": ec,
        }
        logger.info(f"  [{strat_name}] Sharpe={sub_results[strat_name]['sharpe']:.2f}, "
                    f"Return={sub_results[strat_name]['total_return']:.2%}, "
                    f"Trades={trade.get('total_trades', 0)}")

    # 上层资本分配：等权 + 波动率调整
    _alloc = _allocate_strategy_capital(sub_results)

    # 合并 equity_curve
    _merge_results(sub_results, _alloc, trade_log, equity_curve)

    return sub_results


def _rerank_scores(df: pd.DataFrame, factor_weights: dict[str, float]) -> None:
    """用给定子策略权重重新计算综合分析评分。"""
    col_map = {
        "macd": "MACD评分", "momentum": "动量评分", "moneyflow": "资金流评分",
        "quality": "基本面评分", "valuation": "估值评分",
        "top_trader": "龙虎榜评分",
    }

    total_w = sum(factor_weights.values()) or 1.0
    composite = pd.Series(0.0, index=df.index)

    for key, col in col_map.items():
        w = factor_weights.get(key, 0.0)
        if w > 0 and col in df.columns:
            composite += pd.to_numeric(df[col], errors="coerce").fillna(0) * w

    composite /= total_w
    raw = composite
    rng = raw.max() - raw.min()
    df["综合分析评分"] = ((raw - raw.min()) / (rng if rng > 1e-10 else 1) * 100).clip(0, 100)


def _allocate_strategy_capital(
    sub_results: dict[str, dict],
    target_vol: float = 0.15,
) -> dict[str, float]:
    """波动率平价分配：各子策略的波动率贡献相等。"""
    vols = {}
    for name, res in sub_results.items():
        ec = res.get("equity_curve", [])
        if len(ec) < 5:
            vols[name] = 0.20
            continue
        returns = pd.Series([e["portfolio_value"] for e in ec]).pct_change().dropna()
        vols[name] = max(returns.std() * np.sqrt(244), 0.01)  # A股年化交易日244

    inv_vol = {k: 1.0 / v for k, v in vols.items()}
    total = sum(inv_vol.values()) or 1.0
    alloc = {k: v / total for k, v in inv_vol.items()}
    logger.info(f"  策略资本分配: {alloc}")
    return alloc


def _merge_results(
    sub_results: dict[str, dict],
    alloc: dict[str, float],
    merged_trade_log: list[dict[str, Any]],
    merged_equity_curve: list[dict[str, Any]],
) -> None:
    """合并子策略的 trade_log 和 equity_curve。"""
    # 找所有交易日
    all_dates: set[str] = set()
    for res in sub_results.values():
        for e in res.get("equity_curve", []):
            dt = e.get("time")
            if dt is not None:
                all_dates.add(str(dt))

    if not all_dates:
        return

    # 按日期合并 equity_curve
    for dt_str in sorted(all_dates):
        total_val = 0.0
        total_turnover = 0.0
        for name, res in sub_results.items():
            w = alloc.get(name, 0.0)
            for e in res.get("equity_curve", []):
                if str(e.get("time")) == dt_str:
                    total_val += e.get("portfolio_value", 0) * w
                    total_turnover += e.get("turnover", 0) * w
                    break
        merged_equity_curve.append({
            "time": dt_str,
            "portfolio_value": round(total_val, 2),
            "turnover": round(total_turnover, 6),
        })

    # 合并 trade_log
    for name, res in sub_results.items():
        w = alloc.get(name, 0.0)
        for t in res.get("trade_log", []):
            t["strategy"] = name
            t["alloc_weight"] = round(w, 4)
            merged_trade_log.append(t)
