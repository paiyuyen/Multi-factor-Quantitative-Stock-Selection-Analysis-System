"""
因子风险模型 — 组合风险归因模块

基于正交化后的因子暴露矩阵 X_orth（列间相关系数 ≈ 单位阵）与组合权重，
将组合风险拆解为「因子风险」与「个股特质风险」两部分：

    h = X_orthᵀ · w            （组合的因子加权暴露，K 维向量）
    σ²_p = hᵀ · Σ_f · h  +  Σ_i w_i² · σ²_idio,i

其中：
  - Σ_f：因子协方差矩阵。正交化后因子互不相关，默认取单位阵；
    亦可传入估计得到的因子协方差矩阵（含风险因子尺度）。
  - σ_idio,i：个股特质日波动。未提供时用常数估计（并标记 estimated=True）。

输出：
  - 组合日/年化波动率
  - 因子风险占比 vs 特质风险占比
  - 逐因子边际风险贡献（暴露 → 波动贡献 → 占比）
  - 个股特质风险 Top N（含权重与贡献）

与 FactorOrthogonalizer 联动：
    orth = FactorOrthogonalizer()
    result = orth.run(panel, kline, factor_cols)
    risk = FactorRiskModel().from_orthogonalizer(result, weights=w)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_IDIO_VOL = 0.02  # 日波动 2%（≈ 年化 31.7%），未提供特质波动时的常数估计


class FactorRiskModel:
    """基于正交化因子暴露的组合风险归因模型。"""

    DEFAULT_IDIO_VOL = DEFAULT_IDIO_VOL
    TRADING_DAYS = 244  # A股实际年化交易日数均值（非美股252）

    # ── 核心归因 ────────────────────────────────────────────

    def decompose(
        self,
        X_orth: pd.DataFrame,
        weights: pd.Series | None = None,
        factor_cov: pd.DataFrame | np.ndarray | None = None,
        idio_vol: pd.Series | None = None,
        trading_days: int = TRADING_DAYS,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """对正交化因子暴露做风险归因。

        Args:
            X_orth: N×K 正交化因子暴露矩阵，index=股票代码。
            weights: 组合权重 Series（index=股票代码），None = 等权；
                权重会自动归一化为合计 1。
            factor_cov: K×K 因子协方差矩阵（正交化后默认单位阵）。
            idio_vol: 个股特质日波动 Series（index=股票代码），
                None = 常数估计（DEFAULT_IDIO_VOL）。
            trading_days: 年化交易日数。
            top_n: 个股特质风险 Top N 输出数量。

        Returns:
            dict: {
                "组合日波动": float, "组合年化波动": float,
                "因子风险占比": float, "特质风险占比": float,
                "因子风险贡献": pd.DataFrame,
                "个股特质风险TopN": pd.DataFrame,
                "特质波动为估计": bool,
            }
            X_orth 为空或无有效权重时返回 {"error": "..."}。
        """
        if X_orth is None or X_orth.empty:
            return {"error": "因子暴露矩阵为空"}

        X = X_orth.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        if X.empty:
            return {"error": "因子暴露无有效行"}
        k = X.shape[1]

        w = self._normalize_weights(X, weights)
        if w is None:
            return {"error": "无有效权重（全部权重非正或不在暴露矩阵中）"}

        # 因子协方差矩阵：正交化因子默认单位阵
        if factor_cov is None:
            cov = np.eye(k)
        else:
            cov = np.asarray(factor_cov, dtype=float)
            if cov.shape != (k, k):
                return {"error": f"因子协方差矩阵维度应为 ({k},{k})"}

        # 个股特质波动：未提供 → 常数估计
        est_idio = idio_vol is None
        if idio_vol is None:
            sigma_idio = pd.Series(self.DEFAULT_IDIO_VOL, index=X.index)
        else:
            sigma_idio = pd.to_numeric(idio_vol, errors="coerce").reindex(X.index)

        h = X.to_numpy().T @ w.to_numpy()  # K 维组合因子暴露
        factor_var = float(h @ cov @ h)
        idio_var = float(np.sum((w.to_numpy() ** 2) * (sigma_idio.fillna(self.DEFAULT_IDIO_VOL).to_numpy() ** 2)))
        total_var = factor_var + idio_var
        if total_var <= 1e-16:
            return {"error": "组合风险方差为 0（所有波动为 0）"}

        total_vol = float(np.sqrt(total_var))

        # 因子边际贡献（协方差 Σ_f 为一般矩阵时按 2 倍交叉项计入）
        cov_h = cov @ h
        factor_contrib = h * cov_h / total_var  # 逐因子占比，合计 = 因子风险占比
        factor_df = pd.DataFrame(
            {
                "因子": X.columns,
                "组合暴露": np.round(h, 6),
                "风险贡献": np.round(h * cov_h, 8),
                "风险占比": np.round(factor_contrib, 6),
            }
        )
        factor_df = factor_df.sort_values("风险占比", ascending=False).reset_index(drop=True)

        # 个股特质风险贡献
        idio_contrib = (w.to_numpy() ** 2) * (sigma_idio.fillna(self.DEFAULT_IDIO_VOL).to_numpy() ** 2) / total_var
        stock_df = pd.DataFrame(
            {
                "股票": X.index,
                "权重": np.round(w.to_numpy(), 6),
                "特质波动(日)": np.round(sigma_idio.fillna(self.DEFAULT_IDIO_VOL).to_numpy(), 6),
                "特质风险贡献": np.round(idio_contrib, 8),
                "特质风险占比": np.round(idio_contrib, 6),
            }
        ).sort_values("特质风险占比", ascending=False)
        stock_df = stock_df.head(max(int(top_n), 1)).reset_index(drop=True)

        factor_share = float(factor_var / total_var)
        result = {
            "组合日波动": total_vol,
            "组合年化波动": total_vol * np.sqrt(trading_days),
            "因子风险占比": factor_share,
            "特质风险占比": 1.0 - factor_share,
            "因子风险贡献": factor_df,
            "个股特质风险TopN": stock_df,
            "特质波动为估计": est_idio,
        }

        logger.info(
            f"[因子风险] 组合日波动 {result['组合日波动']:.4%} | "
            f"因子风险 {result['因子风险占比']:.1%} / "
            f"特质风险 {result['特质风险占比']:.1%}"
        )
        return result

    @staticmethod
    def _normalize_weights(
        X_orth: pd.DataFrame, weights: pd.Series | None
    ) -> pd.Series | None:
        """对齐并归一化权重：只保留暴露矩阵内且权重为正的股票，合计 = 1。"""
        if weights is None:
            return pd.Series(1.0 / len(X_orth), index=X_orth.index)

        w = pd.to_numeric(weights, errors="coerce")
        w = w[w.notna() & (w > 0)]
        w = w[w.index.isin(X_orth.index)]
        w = w.reindex(X_orth.index).fillna(0.0)
        total = float(w.sum())
        if total <= 1e-12:
            return None
        return w / total

    # ── 与正交化器联动 ──────────────────────────────────────

    def from_orthogonalizer(
        self,
        orth_result: dict[str, Any],
        weights: pd.Series | None = None,
        factor_cov: pd.DataFrame | np.ndarray | None = None,
        idio_vol: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """直接消费 FactorOrthogonalizer().run() 的结果做风险归因。

        Args:
            orth_result: FactorOrthogonalizer.run() 返回的 dict，
                需含 orth_result["orthogonalized"]["X_orth_latest"]。
            weights / factor_cov / idio_vol: 同 decompose()。
            **kwargs: 其余参数透传 decompose()。

        Returns:
            同 decompose()。
        """
        if not isinstance(orth_result, dict) or "orthogonalized" not in orth_result:
            return {"error": "输入不是 FactorOrthogonalizer().run() 的结果"}
        orth = orth_result["orthogonalized"]
        X_orth = orth.get("X_orth_latest")
        if X_orth is None or (isinstance(X_orth, pd.DataFrame) and X_orth.empty):
            return {"error": "正交化结果缺少 X_orth_latest"}
        return self.decompose(
            X_orth,
            weights=weights,
            factor_cov=factor_cov,
            idio_vol=idio_vol,
            **kwargs,
        )
