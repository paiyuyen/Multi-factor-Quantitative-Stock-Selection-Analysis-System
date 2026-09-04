"""窗口预检与容错（Precheck）— 测试。

验收: 在测试集上 precheck 能拦截 ≥95% 已知无效输入，并返回明确原因标签。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BackTrading.precheck import (
    PrecheckStatus,
    fill_ohlcv,
    precheck,
)


def _valid_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.maximum(close, 5.0)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "trade_date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "symbol": "sh600000",
        "open": close - 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": vol,
    })


def _with_nan(df: pd.DataFrame, col: str, frac: float, block: int = 1, seed: int = 1) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    idx = np.sort(rng.choice(n, size=int(n * frac), replace=False))
    for i in idx:
        out.loc[i : min(i + block - 1, n - 1), col] = np.nan
    return out


def _with_zero_vol_days(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """每 step 天插入一天"停牌"：零成交 + 价格横盘（真实 A 股停牌形态）。"""
    out = df.copy()
    mask = np.arange(len(out)) % step == 0
    out.loc[mask, "volume"] = 0.0
    prev = out["close"].shift(1)
    out.loc[mask, "close"] = prev[mask].fillna(out.loc[mask, "close"])
    out.loc[mask, "high"] = out.loc[mask, "close"]
    out.loc[mask, "low"] = out.loc[mask, "close"]
    out.loc[mask, "open"] = out.loc[mask, "close"]
    return out


# ── 已知无效输入拦截（验收: ≥95%） ─────────────────────────────────────

INVALID_CASES = [
    # (name, df, 期望非 OK, 期望原因标签)
    ("too_few_rows", _valid_df(n=10), "TOO_FEW_ROWS"),
    ("close_all_nan", _with_nan(_valid_df(), "close", 1.0, block=1), "NAN_TOO_HIGH_CLOSE"),
    ("close_block_nan40pct", _with_nan(_valid_df(), "close", 0.4, block=20), "NAN_TOO_HIGH_CLOSE"),
    ("volume_all_zero", _valid_df().assign(volume=0.0), "ZERO_VOLUME_ALL"),
    ("volume_zero_with_price_move",
     _valid_df().assign(volume=lambda d: np.where(np.arange(len(d)) % 3 == 0, 0.0, d["volume"])),
     "ZERO_VOLUME_PRICE_MOVE"),
    ("missing_volume_col", _valid_df().drop(columns=["volume"]), "MISSING_OHLCV_COLUMNS"),
    ("zero_volume_ratio_high",
     _valid_df().assign(volume=lambda d: np.where(np.arange(len(d)) % 4 == 0, 0.0, d["volume"])),
     "ZERO_VOLUME_RATIO_HIGH"),
    ("suspension_ratio_high",
     _valid_df().assign(volume=0.0).assign(close=lambda d: d["close"].iloc[0]),  # 全零成交+全横盘
     "SUSPENSION_RATIO_HIGH"),
    ("adj_factor_jump",
     # P1-9 审计修复：12% 向上跳变属正常除权。改为向下回溯跳变（数据源断裂/混用）
     _valid_df().assign(adj_factor=np.concatenate([
         np.full(100, 1.0),
         np.full(100, 1.10),  # normal upward
         np.full(100, 1.05)   # backward jump (1.10 → 1.05, < 0.99×previous)
     ])),
     "ADJ_FACTOR_JUMP"),
    ("price_jump_volume_collapse",
     _valid_df().assign(close=lambda d: np.where(np.arange(len(d)) == 150, d["close"] * 1.6, d["close"]))
               .assign(volume=lambda d: np.where(np.arange(len(d)) == 150, 1000.0, d["volume"])),
     "PRICE_JUMP_VOLUME_ANOMALY"),
    ("ipo_early_spike",
     _valid_df().assign(close=lambda d: np.where(np.arange(len(d)) == 1, d["close"] * 1.5, d["close"])),
     "IPO_EARLY_VOLATILITY"),
    ("empty_series", pd.DataFrame(), "EMPTY_SERIES"),
    ("nan_fillable_scattered", _with_nan(_valid_df(), "close", 0.08, block=1, seed=5),
     "NAN_FILLABLE_CLOSE"),
]

VALID_CASES = [
    ("normal", _valid_df(), PrecheckStatus.OK, []),
    ("normal_with_noise", _valid_df(seed=9), PrecheckStatus.OK, []),
    ("normal_low_vol", _valid_df(seed=3).assign(volume=2_000_000.0), PrecheckStatus.OK, []),
]


@pytest.mark.parametrize("name,df,label", INVALID_CASES, ids=[c[0] for c in INVALID_CASES])
def test_invalid_interception(name, df, label) -> None:
    """已知无效输入必须被拦截（status != OK）且返回明确原因标签。"""
    result = precheck(df)
    assert result.status != PrecheckStatus.OK, f"{name}: 应被拦截，实际 {result.status}"
    assert label in result.reasons, f"{name}: 原因标签 {label} 缺失，实际 {result.reasons}"
    assert isinstance(result.metrics, dict)


def test_acceptance_intercept_rate() -> None:
    """验收: 已知无效输入拦截率 ≥ 95%。"""
    intercepted = sum(1 for _, df, _ in INVALID_CASES if precheck(df).status != PrecheckStatus.OK)
    rate = intercepted / len(INVALID_CASES)
    assert rate >= 0.95, f"拦截率 {rate:.0%} < 95%"


@pytest.mark.parametrize("name,df,status,reasons", VALID_CASES, ids=[c[0] for c in VALID_CASES])
def test_valid_series_pass(name, df, status, reasons) -> None:
    result = precheck(df)
    assert result.status == status, f"{name}: 期望 {status.value}，实际 {result.status} / {result.reasons}"
    assert all(r not in result.reasons for r in reasons)


# ── 模式语义 ───────────────────────────────────────────────────────────

def test_mode_relax_soft_issue_allowed() -> None:
    df = _with_zero_vol_days(_valid_df(), step=4)
    assert precheck(df, mode="RELAX").status == PrecheckStatus.LOW_CONFIDENCE


def test_mode_strict_escalates_to_skip() -> None:
    df = _with_zero_vol_days(_valid_df(), step=4)
    assert precheck(df, mode="STRICT").status == PrecheckStatus.SKIP
    df2 = _with_nan(_valid_df(), "close", 0.08, block=1, seed=5)  # 可修复缺口
    assert precheck(df2, mode="STRICT").status == PrecheckStatus.SKIP


def test_mode_off_bypasses() -> None:
    df = _valid_df().assign(volume=0.0)
    assert precheck(df, mode="OFF").status == PrecheckStatus.OK
    assert precheck(df, mode="OFF").reasons == []


def test_params_dict_override_thresholds() -> None:
    df = _with_zero_vol_days(_valid_df(), step=4)  # 25% 停牌/零成交
    # 放宽阈值 → 不再命中任何检查
    result = precheck(df, params={"max_zero_volume_ratio": 0.4, "max_suspension_ratio": 0.4})
    assert result.status == PrecheckStatus.OK


# ── 漏采 vs 停牌：交叉验证 + 缺失明细 ────────────────────────────────────

def test_suspension_skip_metric_carries_missing_days() -> None:
    df = _valid_df()
    res = precheck(df, {"mode": "RELAX"}, suspension_stats={
        "span_trading_days": 10,
        "suspended_days": ["2024-01-04", "2024-01-05", "2024-01-06"],
        "missing_blocks": [{"start": "2024-01-04", "end": "2024-01-06", "days": 3}],
        "interior_missing_days": ["2024-01-04", "2024-01-05", "2024-01-06"],
        "tail_missing_days": ["2024-01-07"],
    })
    assert res.status == PrecheckStatus.SKIP
    m = res.metrics["SUSPENSION_RATIO_HIGH_CAL"]
    assert m["missing_days"] == ["2024-01-04", "2024-01-05", "2024-01-06"]
    assert m["tail_missing_days"] == ["2024-01-07"]
    assert m["missing_blocks"] == [{"start": "2024-01-04", "end": "2024-01-06", "days": 3}]


def test_cross_validated_under_collection_downgrades_to_low_confidence() -> None:
    """高缺失占比但确认停牌占比低 → 漏采嫌疑，不硬拒（LOW_CONFIDENCE）。"""
    df = _valid_df()
    res = precheck(df, {"mode": "RELAX"}, suspension_stats={
        "span_trading_days": 10,
        "suspended_days": ["2024-01-04", "2024-01-05", "2024-01-06"],  # 30% 缺失
        "cross_validated": True,
        "confirmed_days": ["2024-01-05"],            # 确认停牌仅 10%
        "under_collected_days": ["2024-01-04", "2024-01-06"],  # 漏采嫌疑 20%
        "suspension_ratio_confirmed": 0.1,
        "under_collection_ratio": 0.2,
    })
    assert res.status == PrecheckStatus.LOW_CONFIDENCE
    assert "UNDER_COLLECTION_SUSPECTED" in res.reasons
    assert "SUSPENSION_RATIO_HIGH_CAL" not in res.reasons


def test_cross_validated_confirmed_high_still_skips() -> None:
    """确认停牌占比也超阈值 → 真实停牌，维持硬拒。"""
    df = _valid_df()
    res = precheck(df, {"mode": "RELAX"}, suspension_stats={
        "span_trading_days": 10,
        "suspended_days": ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"],
        "cross_validated": True,
        "confirmed_days": ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07"],
        "under_collected_days": [],
        "suspension_ratio_confirmed": 0.4,
        "under_collection_ratio": 0.0,
    })
    assert res.status == PrecheckStatus.SKIP
    assert "SUSPENSION_RATIO_HIGH_CAL" in res.reasons


def test_confirmed_suspension_days_param_direct() -> None:
    """直接传确认停牌集合（未预计算 cross_validated）也能交叉验证。"""
    df = _valid_df()
    res = precheck(df, {"mode": "RELAX"},
                   suspension_stats={
                       "span_trading_days": 10,
                       "suspended_days": ["2024-01-04", "2024-01-05", "2024-01-06"],
                   },
                   confirmed_suspension_days={"2024-01-04"})
    assert res.status == PrecheckStatus.LOW_CONFIDENCE
    assert "UNDER_COLLECTION_SUSPECTED" in res.reasons


def test_suspension_suspects_summary() -> None:
    from BackTrading.precheck import suspension_suspects
    stats = {
        "sh600000": {
            "suspension_ratio": 0.5, "suspended_days": ["2024-01-04", "2024-01-05"],
            "interior_missing_days": ["2024-01-04", "2024-01-05"],
            "tail_missing_days": [],
            "cross_validated": False,
        },
        "sh600001": {
            "suspension_ratio": 0.3, "suspended_days": ["2024-01-04"],
            "interior_missing_days": ["2024-01-04"], "tail_missing_days": ["2024-01-05"],
            "cross_validated": True, "confirmed_days": ["2024-01-04"], "under_collected_days": [],
        },
        "sh600002": {"suspension_ratio": 0.05, "suspended_days": []},  # 低于阈值不列
    }
    suspects = suspension_suspects(stats)
    assert [s["symbol"] for s in suspects] == ["sh600000", "sh600001"]  # 按占比降序
    assert suspects[0]["ratio"] == 0.5
    assert suspects[1]["cross_validated"] is True
    assert suspects[1]["under_collected_days"] == []


def test_apply_precheck_skip_logs_missing_days(loguru_sink) -> None:
    from BackTrading.precheck import apply_precheck
    out, res = apply_precheck("sh600000", _valid_df(), context="test",
                              suspension_stats={
                                  "span_trading_days": 10,
                                  "suspended_days": ["2024-01-04", "2024-01-05", "2024-01-06"],
                                  "interior_missing_days": ["2024-01-04", "2024-01-05", "2024-01-06"],
                                  "tail_missing_days": [],
                              })
    assert out.empty
    assert "SUSPENSION_RATIO_HIGH_CAL" in res.reasons
    assert any("缺失交易日 3 天" in r for r in loguru_sink)
    assert any("人工复核" in r for r in loguru_sink)


def test_log_suspension_suspects_writes_quality_log() -> None:
    from unittest.mock import MagicMock

    from DataManager.DataQualityChecker import DataQualityChecker
    from BackTrading.precheck import suspension_suspects

    stats = {
        "sh600000": {
            "suspension_ratio": 0.5, "suspended_days": ["2024-01-04", "2024-01-05"],
            "interior_missing_days": ["2024-01-04", "2024-01-05"],
            "tail_missing_days": [], "cross_validated": True,
            "confirmed_days": ["2024-01-04"], "under_collected_days": ["2024-01-05"],
        },
    }
    suspects = suspension_suspects(stats)
    conn = MagicMock()
    eng = MagicMock()
    eng.begin.return_value.__enter__.return_value = conn
    checker = DataQualityChecker(db_engine=eng)
    n = checker.log_suspension_suspects(suspects, "2024-03-31")
    assert n == 1
    assert conn.execute.call_count == 1
    args = conn.execute.call_args[0][1]
    assert args["cn"] == "停牌-疑似漏采"
    assert args["st"] == "warn"
    assert "sh600000" in args["dt"] and "漏采嫌疑=1天" in args["dt"]
    # 无引擎 → 不抛异常，返回 0
    assert DataQualityChecker(db_engine=None).log_suspension_suspects(suspects, "2024-03-31") == 0


# ── 填充容错 ───────────────────────────────────────────────────────────

def test_fill_ohlcv_bounded_gap() -> None:
    df = _valid_df(n=100)
    df.loc[50, "close"] = np.nan
    df.loc[60:80, "close"] = np.nan  # 21 天缺口 > 2×max_fill_gap=10，中间段保持 NaN
    filled = fill_ohlcv(df)
    assert not np.isnan(filled.loc[50, "close"])      # 单日缺口已填充
    assert not np.isnan(filled.loc[60, "close"])      # 缺口前 5 天 ffill
    assert not np.isnan(filled.loc[80, "close"])      # 缺口后 5 天 bfill
    assert np.isnan(filled.loc[70, "close"])          # 缺口中间段保持 NaN
    assert filled.loc[50, "volume"] == df.loc[50, "volume"]  # 非 NaN 列不动


def test_need_fill_returns_filled_frame() -> None:
    from BackTrading.precheck import apply_precheck
    df = _with_nan(_valid_df(), "close", 0.08, block=1, seed=5)
    result = precheck(df)
    assert result.status == PrecheckStatus.NEED_FILL
    filled, res = apply_precheck("sh600000", df, context="test")
    assert res.status == PrecheckStatus.NEED_FILL
    assert not filled["close"].isna().any()
    assert len(filled) == len(df)


# ── 集成：指标计算前的 precheck 拦截 ──────────────────────────────────

def test_precompute_skip_and_snapshot(tmp_path, monkeypatch) -> None:
    """precompute_all_indicators: SKIP 股票跳过计算 + 失败快照（precheck_status 入 schema）。"""
    from BackTrading.indicator_cache import (
        precompute_all_indicators, _IN_MEMORY, _reset_memory_caches,
    )
    from BackTrading.snapshot import find_snapshots

    _reset_memory_caches()
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    _valid_df(n=300).to_parquet(stock_dir / "sh600000.parquet", index=False)      # 正常
    _valid_df(n=300).assign(volume=0.0).to_parquet(stock_dir / "sz000001.parquet", index=False)  # 全零成交

    precompute_all_indicators(str(stock_dir))
    assert len(_IN_MEMORY["sh600000"]) == 300
    assert len(_IN_MEMORY["sz000001"]) == 0  # SKIP → 空帧占位

    snaps = find_snapshots()
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap.error_code == "PRECHECK_SKIP"
    assert snap.symbol == "sz000001"
    assert snap.precheck_status is not None
    assert snap.precheck_status["status"] == "SKIP"
    assert "ZERO_VOLUME_ALL" in snap.precheck_status["reasons"]


def test_precompute_need_fill_records_reasons(tmp_path) -> None:
    """precompute_all_indicators: NEED_FILL 股票填充后计算，meta 记录 precheck 原因。"""
    from BackTrading.indicator_cache import (
        precompute_all_indicators, _IN_MEMORY, _meta_path, _reset_memory_caches,
    )
    import json as _json

    _reset_memory_caches()
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    _with_nan(_valid_df(n=300), "close", 0.08, block=1, seed=5).to_parquet(
        stock_dir / "sh600000.parquet", index=False)

    precompute_all_indicators(str(stock_dir))
    assert len(_IN_MEMORY["sh600000"]) == 300  # 填充后照常计算
    meta = _json.loads(_meta_path("sh600000").read_text(encoding="utf-8"))
    assert meta.get("precheck") == ["NAN_FILLABLE_CLOSE"]


def test_stock_worker_skip_returns_empty(tmp_path, monkeypatch) -> None:
    """_stock_worker_vectorized: SKIP 股票返回 []（不进入信号计算）。

    P0-10 ②：循环路径 _stock_worker 已删除，SKIP 拦截在 Phase 0 预计算阶段。
    """
    import BackTrading.indicator_cache as ic
    monkeypatch.setattr(ic, "_cache_root", lambda: tmp_path / "icache")
    from BackTrading.prepare import _stock_worker_vectorized, precompute_all_indicators

    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    _valid_df(n=300).assign(volume=0.0).to_parquet(stock_dir / "sh600000.parquet", index=False)
    precompute_all_indicators(str(stock_dir), shard_mode="off")
    rows = _stock_worker_vectorized("sh600000", str(stock_dir), {})
    assert rows == []


def test_precheck_summary_counts() -> None:
    from BackTrading.precheck import precheck_summary
    pool = pd.concat([
        _valid_df(n=300, seed=1).assign(symbol="sh600000"),
        _valid_df(n=300, seed=2).assign(symbol="sh600001").assign(volume=0.0),
        _valid_df(n=300, seed=3).assign(symbol="sh600002"),
    ], ignore_index=True)
    counts = precheck_summary(pool)
    assert counts["OK"] == 2
    assert counts["SKIP"] == 1
