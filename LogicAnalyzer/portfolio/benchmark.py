"""
基准对比评估器

将组合持仓的加权收益率与基准指数对比，计算：
  - 累计收益率（组合 vs 基准）
  - Alpha（超额收益）
  - Beta（市场敏感度）
  - 夏普比率
  - 最大回撤
  - 跟踪误差
  - 信息比率
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class BenchmarkEvaluator:
    """将组合收益率与基准指数进行对比分析。"""

    def __init__(self, config: Any) -> None:  # noqa: ANN401
        self.config = config

    def evaluate(self, portfolio_returns: pd.Series | None = None,
                 benchmark_returns: pd.Series | None = None,
                 portfolio_df: pd.DataFrame | None = None,
                 benchmark_df: pd.DataFrame | None = None,
                 date_col: str = "trade_date",
                 ret_col: str = "daily_return",
                 risk_free_rate: float = 0.02) -> dict[str, Any]:
        """执行基准对比评估。

        Args:
            portfolio_returns: 组合日收益率 Series（index=date）。
            benchmark_returns: 基准日收益率 Series（index=date）。
            portfolio_df: 组合日线 DataFrame（含日期列和收益率列）。
            benchmark_df: 基准日线 DataFrame（含日期列和收益率列）。
            date_col: 日期列名。
            ret_col: 收益率列名。
            risk_free_rate: 无风险利率（年化，默认 2%）。

        Returns:
            dict: {
                "累计收益_组合": float,
                "累计收益_基准": float,
                "超额收益_Alpha": float,
                "Beta": float,
                "夏普比率_年化": float,
                "最大回撤_组合": float,
                "最大回撤_基准": float,
                "跟踪误差_年化": float,
                "信息比率": float,
                "收益率序列": pd.DataFrame,
            }
        """
        # 从 DataFrame 提取收益率序列
        if portfolio_returns is None and portfolio_df is not None:
            if ret_col in portfolio_df.columns:
                pf = portfolio_df[[date_col, ret_col]].copy()
                pf[date_col] = pd.to_datetime(pf[date_col])
                pf = pf.set_index(date_col).sort_index()
                portfolio_returns = pf[ret_col]

        if benchmark_returns is None and benchmark_df is not None:
            if ret_col in benchmark_df.columns:
                bm = benchmark_df[[date_col, ret_col]].copy()
                bm[date_col] = pd.to_datetime(bm[date_col])
                bm = bm.set_index(date_col).sort_index()
                benchmark_returns = bm[ret_col]

        if portfolio_returns is None or benchmark_returns is None:
            logger.warning("[BenchmarkEval] 缺少组合或基准收益率数据")
            return {"error": "数据不足"}

        # 对齐日期
        combined = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }).dropna()
        if len(combined) < 2:
            logger.warning("[BenchmarkEval] 对齐后数据不足 2 期")
            return {"error": "数据不足"}

        p_ret = combined["portfolio"]
        b_ret = combined["benchmark"]

        # 累计收益
        cum_p = (1 + p_ret).prod() - 1
        cum_b = (1 + b_ret).prod() - 1

        # 日收益率均值/标准差
        mean_p = p_ret.mean()
        mean_b = b_ret.mean()
        std_p = p_ret.std()

        # Alpha = 组合收益 - Beta × 基准收益
        cov = p_ret.cov(b_ret)
        var_b = b_ret.var()
        beta = cov / var_b if var_b > 0 else 1.0

        # 年化因子（A股实际年化交易日数均值，非美股252）
        _TRADING_DAYS = 244
        n = len(combined)
        ann_factor = _TRADING_DAYS / n if n > 0 else _TRADING_DAYS
        ann_p = (1 + mean_p) ** _TRADING_DAYS - 1
        ann_b = (1 + mean_b) ** _TRADING_DAYS - 1

        # Alpha（年化）
        alpha = (ann_p - risk_free_rate) - beta * (ann_b - risk_free_rate)

        # 夏普比率（年化）
        excess_p = mean_p - risk_free_rate / _TRADING_DAYS
        sharpe = (excess_p / std_p * np.sqrt(_TRADING_DAYS)) if std_p > 0 else 0

        # 最大回撤
        max_dd_p = self._max_drawdown(p_ret)
        max_dd_b = self._max_drawdown(b_ret)

        # 跟踪误差（年化）
        diff = p_ret - b_ret
        tracking_error = diff.std() * np.sqrt(_TRADING_DAYS)

        # 信息比率
        info_ratio = (mean_p - mean_b) / diff.std() if diff.std() > 0 else 0

        result = {
            "累计收益_组合": cum_p,
            "累计收益_基准": cum_b,
            "超额收益_Alpha": alpha,
            "Beta": beta,
            "夏普比率_年化": sharpe,
            "最大回撤_组合": max_dd_p,
            "最大回撤_基准": max_dd_b,
            "跟踪误差_年化": tracking_error,
            "信息比率": info_ratio,
            "日收益率数据": combined,
            "总交易日": n,
        }

        logger.info(
            f"[BenchmarkEval] "
            f"组合累计 {cum_p:.2%} vs 基准 {cum_b:.2%} | "
            f"Alpha {alpha:.2%} | Beta {beta:.2f} | "
            f"Sharpe {sharpe:.2f} | 最大回撤 {max_dd_p:.2%}"
        )
        return result

    @staticmethod
    def estimate_portfolio_returns(portfolio_df: pd.DataFrame,
                                   kline_df: pd.DataFrame,
                                   weight_col: str = "目标权重",
                                   code_col: str = "股票代码") -> pd.Series:
        """估算组合历史日收益率（按目标权重加权）。

        Args:
            portfolio_df: 当前持仓 DataFrame（含 目标权重）。
            kline_df: 全市场 K 线 DataFrame（含 symbol, trade_date, close）。
            weight_col: 权重列名。
            code_col: 股票代码列名（纯代码，如 600519）。

        Returns:
            Series: 日收益率序列，index=trade_date。
        """
        if portfolio_df.empty or kline_df.empty:
            return pd.Series(dtype=float)

        # 对齐 symbol 格式：kline_df["symbol"] 是 sh600519，portfolio_df[code_col] 是 600519
        kline = kline_df.copy()
        if "股票代码" not in kline.columns:
            from UtilsManager.CodeNormalizer import CodeNormalizer
            kline["股票代码"] = kline["symbol"].apply(CodeNormalizer.normalize)

        # 过滤有持仓的股票
        weighted = portfolio_df[portfolio_df[weight_col].fillna(0) > 0]
        if weighted.empty:
            return pd.Series(dtype=float)

        # 为每只持仓股票计算日收益率
        all_rets = []
        for _, row in weighted.iterrows():
            symbol = row[code_col]
            weight = row[weight_col]
            stock_kline = kline[kline["股票代码"] == symbol].sort_values("trade_date")
            if len(stock_kline) < 2:
                continue
            stock_kline["return"] = stock_kline["close"].pct_change()
            stock_kline["weighted_return"] = stock_kline["return"] * weight
            all_rets.append(stock_kline[["trade_date", "weighted_return"]])

        if not all_rets:
            return pd.Series(dtype=float)

        combined = pd.concat(all_rets)
        portfolio_rets = combined.groupby("trade_date")["weighted_return"].sum()
        return portfolio_rets.sort_index()

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> float:
        """计算最大回撤。"""
        cum = (1 + returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return dd.min()
