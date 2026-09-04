from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from LogicAnalyzer.risk.historical_var import HistoricalVaR


def _returns(n: int = 5000, seed: int = 0, mu: float = 0.0005, sigma: float = 0.02) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(mu, sigma, n),
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def test_compute_var_matches_normal_quantile() -> None:
    res = HistoricalVaR().compute(_returns())
    assert "error" not in res
    # 正态收益下 95% VaR ≈ 1.645σ - μ（正数 = 损失）
    assert res["VaR_95%"] == pytest.approx(1.645 * 0.02 - 0.0005, rel=0.05)
    assert res["VaR_99%"] > res["VaR_95%"]


def test_es_geq_var() -> None:
    res = HistoricalVaR().compute(_returns())
    assert res["ES_95%"] >= res["VaR_95%"]
    assert res["ES_99%"] >= res["VaR_99%"]


def test_window_limits_to_recent() -> None:
    r = _returns()
    res = HistoricalVaR().compute(r, window=100)
    assert res["样本数"] == 100
    manual = HistoricalVaR().compute(r.tail(100))
    assert res["VaR_95%"] == manual["VaR_95%"]


def test_horizon_scaling() -> None:
    res5 = HistoricalVaR().compute(_returns(), horizon_days=5)
    res1 = HistoricalVaR().compute(_returns())
    # 结果四舍五入到 6 位小数，放宽到 1e-4 相对误差
    assert res5["VaR_95%"] == pytest.approx(res1["VaR_95%"] * np.sqrt(5), rel=1e-4)


def test_annualized_vol() -> None:
    r = _returns(sigma=0.03)
    res = HistoricalVaR().compute(r)
    # 样本标准差存在抽样误差（n=5000，se≈σ/√(2n)≈0.03%），放宽到 1% 相对误差
    assert res["年化波动"] == pytest.approx(0.03 * np.sqrt(244), rel=0.01)


def test_rolling_series() -> None:
    df = HistoricalVaR().rolling_series(_returns(n=300), confidence=0.95, window=60, min_periods=30)
    assert list(df.columns) == ["VaR", "ES", "滚动波动"]
    assert df["VaR"].notna().sum() >= 270
    assert df["VaR"].iloc[-1] > 0


def test_clean_removes_inf_and_nan() -> None:
    r = _returns(n=100)
    r.iloc[0] = np.inf
    r.iloc[1] = np.nan
    res = HistoricalVaR().compute(r)
    assert res["样本数"] == 98


def test_empty_returns_error() -> None:
    res = HistoricalVaR().compute(pd.Series(dtype=float))
    assert "error" in res


def test_too_few_samples_error() -> None:
    res = HistoricalVaR().compute(pd.Series([0.01]))
    assert "error" in res


def test_build_report_rows_and_columns() -> None:
    df = HistoricalVaR().build_report(_returns())
    assert list(df.columns) == ["置信度", "VaR(日)", "ES(日)", "VaR(5日)", "ES(5日)"]
    assert len(df) == 4  # 95% / 99% + 总览 + 样本数
    assert (df["置信度"] == "95%").any()
    assert (df["置信度"] == "99%").any()


def test_build_report_es_positive() -> None:
    df = HistoricalVaR().build_report(_returns())
    vals = df[df["置信度"].isin(["95%", "99%"])]
    assert (pd.to_numeric(vals["ES(日)"], errors="coerce") > 0).all()
