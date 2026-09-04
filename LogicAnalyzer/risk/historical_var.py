"""
历史模拟法 VaR / ES 风险度量模块

对组合（或单资产）日收益率序列做尾部风险度量：
  - VaR（Value at Risk）：历史收益分布的分位数。95% VaR 表示有 95% 概率
    日损失不超过该值（输出为正数，即损失金额/比例）。
  - ES（Expected Shortfall，期望损失）：超出 VaR 阈值部分的尾部样本均值，
    衡量"亏穿 VaR"之后的平均损失。

支持：
  - 单期计算：多置信度 VaR / ES + 年化波动率
  - 滚动窗口计算（window 参数截取最近 N 个交易日）
  - 滚动 VaR / ES 序列（时序预警用）
  - 汇总报告表（供 Excel 报告直接写入）

用法:
    var = HistoricalVaR()
    result = var.compute(portfolio_returns)
    # result["VaR_95%"] = 0.0213   # 日 VaR（正数表示损失）
    # result["ES_99%"]  = 0.0351   # 尾部期望损失
    sheet = var.build_report(portfolio_returns)  # DataFrame，可直接进 Excel
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_CONFIDENCE: tuple[float, ...] = (0.95, 0.99)


class HistoricalVaR:
    """历史模拟法 VaR / ES 计算器。"""

    DEFAULT_CONFIDENCE = DEFAULT_CONFIDENCE
    TRADING_DAYS = 244  # A股实际年化交易日数均值（非美股252）

    # ── 数据清洗 ────────────────────────────────────────────

    @staticmethod
    def _clean_returns(returns: pd.Series) -> pd.Series:
        """剔除缺失与无穷值，返回数值型日收益率序列。"""
        clean = pd.to_numeric(returns, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        return clean

    # ── 单期计算 ────────────────────────────────────────────

    def compute(
        self,
        returns: pd.Series,
        confidence_levels: tuple[float, ...] | list[float] | None = None,
        window: int | None = None,
        horizon_days: int = 1,
        trading_days: int = TRADING_DAYS,
    ) -> dict[str, Any]:
        """计算历史模拟法 VaR 与 ES。

        Args:
            returns: 组合日收益率 Series（index=日期）。
            confidence_levels: 置信度序列，默认 (0.95, 0.99)。
            window: 仅使用最近 N 个交易日（None = 全部历史）。
            horizon_days: 持有期（天），按 sqrt(h) 缩放 VaR/ES（iid 假设）。
            trading_days: 年化交易日数。

        Returns:
            dict: {
                "样本数": int,
                "日均收益": float,
                "年化波动": float,
                "VaR_95%": float, "ES_95%": float,
                "VaR_99%": float, "ES_99%": float,
                "VaR_5日": float,  # horizon_days > 1 时附带
            }
            数据不足时返回 {"error": "..."}。
        """
        clean = self._clean_returns(returns)
        if clean.empty:
            logger.warning("[VaR] 收益率序列为空，无法计算")
            return {"error": "收益率序列为空"}

        if window is not None and window > 0:
            clean = clean.tail(int(window))
        if len(clean) < 2:
            logger.warning("[VaR] 有效样本不足 2 期")
            return {"error": "有效样本不足"}

        levels = tuple(confidence_levels or self.DEFAULT_CONFIDENCE)
        scale = float(np.sqrt(max(int(horizon_days), 1)))

        result: dict[str, Any] = {
            "样本数": int(len(clean)),
            "日均收益": float(clean.mean()),
            "年化波动": float(clean.std() * np.sqrt(trading_days)),
        }

        for q in levels:
            if not 0 < q < 1:
                continue
            var_daily = self._var(clean, q)
            es_daily = self._es(clean, q)
            if horizon_days > 1:
                result[f"VaR_{int(q * 100)}%"] = round(var_daily * scale, 6)
                result[f"ES_{int(q * 100)}%"] = round(es_daily * scale, 6)
            else:
                result[f"VaR_{int(q * 100)}%"] = round(var_daily, 6)
                result[f"ES_{int(q * 100)}%"] = round(es_daily, 6)

        logger.info(
            f"[VaR] 样本 {result['样本数']} | "
            + " | ".join(
                f"VaR{q:.0%} {result.get(f'VaR_{int(q * 100)}%', 0):.4f} / "
                f"ES{q:.0%} {result.get(f'ES_{int(q * 100)}%', 0):.4f}"
                for q in levels
            )
        )
        return result

    @staticmethod
    def _var(returns: pd.Series, confidence: float) -> float:
        """历史模拟 VaR：1-confidence 分位数（取正 = 损失量级）。"""
        arr = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
        threshold = float(np.quantile(arr, 1.0 - confidence))
        return float(-threshold)

    @staticmethod
    def _es(returns: pd.Series, confidence: float) -> float:
        """历史模拟 ES：不优于 VaR 阈值（更亏）的尾部样本均值。"""
        arr = returns.to_numpy() if isinstance(returns, pd.Series) else np.asarray(returns)
        threshold = np.quantile(arr, 1.0 - confidence)
        tail = arr[arr <= threshold]
        if len(tail) == 0:
            return 0.0
        return float(-tail.mean())

    # ── 滚动序列 ────────────────────────────────────────────

    def rolling_series(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        window: int = 60,
        min_periods: int = 30,
    ) -> pd.DataFrame:
        """滚动窗口 VaR / ES 序列（时序风险预警用）。

        Args:
            returns: 日收益率 Series。
            confidence: 置信度（默认 95%）。
            window: 滚动窗口长度。
            min_periods: 最少样本数（不足为 NaN）。

        Returns:
            DataFrame: 列 [VaR, ES, 滚动波动]，index=日期。
        """
        clean = self._clean_returns(returns)
        if clean.empty:
            return pd.DataFrame(columns=["VaR", "ES", "滚动波动"])

        rolling = clean.rolling(window=int(window), min_periods=int(min_periods))
        var_series = rolling.apply(lambda x: self._var(x, confidence), raw=True)
        es_series = rolling.apply(lambda x: self._es(x, confidence), raw=True)
        vol_series = rolling.std()

        return pd.DataFrame(
            {
                "VaR": var_series.rename(f"VaR_{int(confidence * 100)}%"),
                "ES": es_series.rename(f"ES_{int(confidence * 100)}%"),
                "滚动波动": vol_series.rename("滚动波动"),
            }
        )

    # ── Excel 报告表 ────────────────────────────────────────

    def build_report(
        self,
        returns: pd.Series,
        confidence_levels: tuple[float, ...] | list[float] | None = None,
        window: int | None = None,
        horizon_days: int = 1,
    ) -> pd.DataFrame:
        """生成适合写入 Excel 的 VaR/ES 汇总表。

        Returns:
            DataFrame: 每行一个置信度，含 VaR / ES（日、5 日）与总览行。
        """
        levels = tuple(confidence_levels or self.DEFAULT_CONFIDENCE)
        clean = self._clean_returns(returns)
        if window is not None and window > 0:
            clean = clean.tail(int(window))

        rows = []
        for q in levels:
            level = int(q * 100)
            var_daily = self._var(clean, q)
            es_daily = self._es(clean, q)
            rows.append(
                {
                    "置信度": f"{level}%",
                    "VaR(日)": round(var_daily, 6),
                    "ES(日)": round(es_daily, 6),
                    "VaR(5日)": round(var_daily * np.sqrt(5), 6),
                    "ES(5日)": round(es_daily * np.sqrt(5), 6),
                }
            )

        result = self.compute(
            returns,
            confidence_levels=levels,
            window=window,
            horizon_days=horizon_days,
        )
        if "error" not in result:
            rows.append(
                {
                    "置信度": "总览",
                    "VaR(日)": "",
                    "ES(日)": "",
                    "VaR(5日)": "",
                    "ES(5日)": "",
                }
            )
            rows.append(
                {
                    "置信度": f"样本数={result['样本数']}",
                    "VaR(日)": "",
                    "ES(日)": "",
                    "VaR(5日)": "",
                    "ES(5日)": "",
                }
            )

        report = pd.DataFrame(rows)
        logger.info(f"[VaR] 报告表生成完成（{len(levels)} 个置信度）")
        return report
