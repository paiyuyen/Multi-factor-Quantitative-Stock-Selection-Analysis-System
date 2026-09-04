"""统计显著性基础校验单元测试"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from LogicAnalyzer.statistical_significance import (
    check_min_trades,
    check_holding_period_health,
    check_market_cycle_coverage,
    run_significance_check,
    MIN_TOTAL_TRADES,
    MIN_AVG_HOLDING_DAYS,
    MIN_DAILY_STRATEGY_WINRATE,
    MIN_MARKET_CYCLE_YEARS,
    MIN_BULL_MARKET_DAYS,
    MIN_BEAR_MARKET_DAYS,
    MAX_CONTINUOUS_UP_DAYS,
)


def _round_dates(n: int, hold_days: int = 1, base: str = "2023-01-01"):
    """生成 n 个回合的确定性买卖日期对（每个回合独立标的）。"""
    base_ts = pd.Timestamp(base)
    for i in range(n):
        buy_ts = base_ts + pd.Timedelta(days=3 * i)
        sell_ts = buy_ts + pd.Timedelta(days=hold_days)
        yield f"STOCK{i:04d}", buy_ts.strftime("%Y-%m-%d"), sell_ts.strftime("%Y-%m-%d")


# ── 1. 最小样本量约束 ──

class TestMinTrades:
    def _make_log(self, n_rounds: int) -> list[dict]:
        log = []
        for sym, buy_date, sell_date in _round_dates(n_rounds):
            log.append({
                "time": buy_date,
                "symbol": sym,
                "action": "buy",
                "value": 10000.0,
                "cost": 10.0,
                "qty": 100,
            })
            log.append({
                "time": sell_date,
                "symbol": sym,
                "action": "sell",
                "value": 10100.0,
                "cost": 10.0,
                "qty": 100,
            })
        return log

    def test_pass_enough_trades(self):
        log = self._make_log(120)
        r = check_min_trades(log)
        assert r.passed is True

    def test_fail_few_trades(self):
        log = self._make_log(50)
        r = check_min_trades(log)
        assert r.passed is False
        assert "统计显著性不足" in r.reason

    def test_fail_empty_log(self):
        r = check_min_trades([])
        assert r.passed is False

    def test_exactly_at_threshold(self):
        log = self._make_log(MIN_TOTAL_TRADES)
        r = check_min_trades(log)
        assert r.passed is True

    def test_partial_sells_not_double_counted(self):
        """部分卖出拆出多条记录，不得虚增回合数。

        55 个回合各含 sell_partial + sell 两条记录（共 110 条卖出），
        回合数应计为 55 < 100 → FAIL；若按卖出条数统计则会误判 PASS。
        """
        log = []
        for sym, buy_date, sell_date in _round_dates(55, hold_days=1):
            log.append({"time": buy_date, "symbol": sym, "action": "buy", "value": 10000.0, "cost": 10.0, "qty": 100})
            log.append({"time": sell_date, "symbol": sym, "action": "sell_partial", "value": 4100.0, "cost": 4.0, "qty": 40})
            log.append({"time": sell_date, "symbol": sym, "action": "sell", "value": 6000.0, "cost": 6.0, "qty": 60})
        r = check_min_trades(log)
        assert r.passed is False
        assert "55" in r.reason

    def test_partial_sells_counted_as_one_round(self):
        log = []
        for sym, buy_date, sell_date in _round_dates(120, hold_days=1):
            log.append({"time": buy_date, "symbol": sym, "action": "buy", "value": 10000.0, "cost": 10.0, "qty": 100})
            log.append({"time": sell_date, "symbol": sym, "action": "sell_partial", "value": 4100.0, "cost": 4.0, "qty": 40})
            log.append({"time": sell_date, "symbol": sym, "action": "sell", "value": 6000.0, "cost": 6.0, "qty": 60})
        r = check_min_trades(log)
        assert r.passed is True


# ── 2. 平均持仓周期健康度 ──

class TestHoldingPeriod:
    def _make_log(self, hold_days: int = 1, win_pct: float = 0.50, n: int = 120):
        """生成含买卖对的交易日志（引擎字段格式: time/qty）。"""
        log = []
        wins = int(n * win_pct)
        for i, (sym, buy_date, sell_date) in enumerate(_round_dates(n, hold_days=hold_days)):
            sell_val = 10200.0 if i < wins else 9800.0
            log.append({
                "time": buy_date,
                "symbol": sym,
                "action": "buy",
                "value": 10000.0,
                "cost": 10.0,
                "qty": 100,
            })
            log.append({
                "time": sell_date,
                "symbol": sym,
                "action": "sell",
                "value": sell_val,
                "cost": 10.0,
                "qty": 100,
            })
        return log

    def test_fail_short_hold_low_winrate(self):
        """持仓<2天 + 胜率<55% → FAIL。"""
        log = self._make_log(hold_days=1, win_pct=0.50)
        r = check_holding_period_health(log)
        assert r.passed is False
        assert "无法实盘盈利" in r.reason

    def test_pass_long_hold(self):
        """持仓较长 → PASS（即使胜率偏低）。"""
        log = self._make_log(hold_days=10, win_pct=0.50)
        r = check_holding_period_health(log)
        assert r.passed is True

    def test_pass_high_winrate(self):
        """胜率高 → PASS（即使持仓短）。"""
        log = self._make_log(hold_days=1, win_pct=0.70)
        r = check_holding_period_health(log)
        assert r.passed is True

    def test_empty_log(self):
        r = check_holding_period_health([])
        assert r.passed is False

    def test_partial_sell_weighted_holding(self):
        """部分卖出按股数加权计持仓天数，同一买入不得被二次匹配。

        买入 100 股 @D1；D3 卖 40 股、D5 卖 60 股 →
        平均持仓 = (2*40 + 4*60) / 100 = 3.2 天 → PASS。
        """
        log = [
            {"time": "2023-01-01", "symbol": "S1", "action": "buy", "value": 10000.0, "cost": 10.0, "qty": 100},
            {"time": "2023-01-03", "symbol": "S1", "action": "sell_partial", "value": 4100.0, "cost": 4.0, "qty": 40},
            {"time": "2023-01-05", "symbol": "S1", "action": "sell", "value": 6120.0, "cost": 6.0, "qty": 60},
        ]
        r = check_holding_period_health(log)
        assert r.passed is True
        assert "3.2" in r.details[0]

    def test_engine_field_time_only(self):
        """仅含引擎 time 字段（无 trade_date）也能正确匹配。"""
        log = self._make_log(hold_days=3, win_pct=0.80)
        r = check_holding_period_health(log)
        assert r.passed is True


# ── 3. 多重牛熊覆盖 ──

class TestMarketCycle:
    def _make_kline(self, start_year: int = 2020, n_years: int = 4, mix: str = "normal"):
        """生成含牛熊交替的 K 线数据。"""
        dates = pd.bdate_range(start=f"{start_year}-01-01", periods=n_years * 244)
        if mix == "normal":
            rets = np.random.RandomState(42).normal(0.0003, 0.015, len(dates))
        elif mix == "only_up":
            rets = np.random.RandomState(42).uniform(0.001, 0.01, len(dates))
        elif mix == "only_down":
            rets = np.random.RandomState(42).uniform(-0.01, -0.001, len(dates))
        prices = 100.0 * np.cumprod(1 + rets)
        df = pd.DataFrame({
            "trade_date": dates.strftime("%Y-%m-%d"),
            "close": prices,
            "symbol": "000001",
        })
        return df

    def test_pass_full_cycle(self):
        df = self._make_kline(start_year=2020, n_years=4, mix="normal")
        r = check_market_cycle_coverage(df)
        assert r.passed is True

    def test_fail_short_span(self):
        df = self._make_kline(start_year=2023, n_years=1, mix="normal")
        r = check_market_cycle_coverage(df)
        assert r.passed is False
        assert "年" in r.reason

    def test_fail_only_up(self):
        df = self._make_kline(start_year=2020, n_years=4, mix="only_up")
        r = check_market_cycle_coverage(df)
        assert r.passed is False
        assert "熊市" in r.reason or "下跌" in r.reason

    def test_fail_only_down(self):
        df = self._make_kline(start_year=2020, n_years=4, mix="only_down")
        r = check_market_cycle_coverage(df)
        assert r.passed is False
        assert "牛市" in r.reason or "上涨" in r.reason

    def test_empty_df(self):
        r = check_market_cycle_coverage(pd.DataFrame())
        assert r.passed is False


# ── 4. 一站式入口 ──

def _significance_log(n_rounds: int, hold_days: int = 28):
    log = []
    for sym, buy_date, sell_date in _round_dates(n_rounds, hold_days=hold_days, base="2020-01-01"):
        log.append({"time": buy_date, "symbol": sym, "action": "buy", "value": 10000.0, "cost": 10.0, "qty": 100})
        log.append({"time": sell_date, "symbol": sym, "action": "sell", "value": 10200.0, "cost": 10.0, "qty": 100})
    return log


def _significance_kline(n_periods: int = 800):
    return pd.DataFrame({
        "trade_date": pd.bdate_range(start=2020, periods=n_periods).strftime("%Y-%m-%d"),
        "close": 100.0 * np.cumprod(1 + np.random.RandomState(42).normal(0.0003, 0.015, n_periods)),
        "symbol": "000001",
    })


class TestSignificanceSummary:
    def test_all_pass(self):
        log = _significance_log(120)
        df = _significance_kline()
        summary = run_significance_check(log, df)
        assert summary.passed is True

    def test_fail_sample_size(self):
        log = _significance_log(50)
        df = _significance_kline()
        summary = run_significance_check(log, df)
        assert summary.min_sample_check.passed is False
        assert summary.passed is False

    def test_fail_holding_period(self):
        """引擎真实场景：持仓 1 天 + 胜率 50% → 综合判定废弃。"""
        log = _significance_log(120, hold_days=1)
        for t in log:
            if t["action"] == "sell":
                t["value"] = 9800.0
        df = _significance_kline()
        summary = run_significance_check(log, df)
        assert summary.holding_period_check.passed is False
        assert summary.passed is False

    def test_trading_day_holding_with_calendar(self):
        """传入交易日历后，持仓天数按交易日（而非自然日）计算。

        周末持仓跨过自然 3 天但仅 1 个交易日 → avg=1 < 2 且胜率不足 → FAIL。
        """
        log = [
            {"time": "2023-01-05", "symbol": "S1", "action": "buy", "value": 10000.0, "cost": 10.0, "qty": 100},
            {"time": "2023-01-09", "symbol": "S1", "action": "sell", "value": 9800.0, "cost": 10.0, "qty": 100},
        ]
        trade_dates = ["2023-01-05", "2023-01-09"]
        r = check_holding_period_health(log, trade_dates)
        assert r.passed is False
        # 无日历退化：自然日差 4 天 ≥ 2 → PASS（验证日历生效差异）
        r_cal = check_holding_period_health(log, None)
        assert r_cal.passed is True
