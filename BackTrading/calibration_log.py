from __future__ import annotations

import json
import math
from datetime import date, datetime
from typing import Any

import numpy as np
from loguru import logger
from sqlalchemy import text

TABLE = "backtest_calibration_log"

# ── 多重测试惩罚配置（Multiple Testing Deception） ──
MAX_TUNING_ATTEMPTS = 10            # 同区间调参尝试上限
MULTIPLE_TESTING_PENALTY = 0.20     # 超限后 Sharpe/Sortino 硬扣减比例


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              SERIAL PRIMARY KEY,
    run_time        TIMESTAMP   NOT NULL DEFAULT NOW(),
    frequency       VARCHAR(16) NOT NULL,
    backtest_start_date VARCHAR(8) NOT NULL,
    out_of_sample_days INT        NOT NULL,
    initial_cash    NUMERIC(14,2) NOT NULL,
    params          JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    sharpe          NUMERIC(8,4),
    sortino         NUMERIC(8,4),
    calmar          NUMERIC(8,4),
    total_return    NUMERIC(8,4),
    annual_return   NUMERIC(8,4),
    annual_vol      NUMERIC(8,4),
    max_drawdown    NUMERIC(8,4),
    max_drawdown_duration INT DEFAULT 0,
    var_95          NUMERIC(8,4),
    cvar_95         NUMERIC(8,4),
    win_rate        NUMERIC(6,4),
    profit_factor   NUMERIC(10,4),
    total_trades    INT DEFAULT 0,
    status          VARCHAR(16) NOT NULL DEFAULT 'success',
    git_commit      VARCHAR(12) DEFAULT '',
    config_hash     VARCHAR(8)  DEFAULT '',
    data_version    VARCHAR(40) DEFAULT '',
    pbo             NUMERIC(6,4) DEFAULT 0.0,
    dsr             NUMERIC(6,4) DEFAULT 0.0,
    num_trials      INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_{TABLE}_run_time ON {TABLE} (run_time DESC);
"""


def ensure_table(engine: Any) -> None:
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
    # 迁移：兼容旧表
    for col, typ in [
        ("lookback_days", None),
        ("sortino", "NUMERIC(8,4)"),
        ("calmar", "NUMERIC(8,4)"),
        ("annual_return", "NUMERIC(8,4)"),
        ("annual_vol", "NUMERIC(8,4)"),
        ("max_drawdown_duration", "INT DEFAULT 0"),
        ("var_95", "NUMERIC(8,4)"),
        ("cvar_95", "NUMERIC(8,4)"),
        ("win_rate", "NUMERIC(6,4)"),
        ("profit_factor", "NUMERIC(10,4)"),
        ("total_trades", "INT DEFAULT 0"),
        ("git_commit", "VARCHAR(12) DEFAULT ''"),
        ("config_hash", "VARCHAR(8) DEFAULT ''"),
        ("data_version", "VARCHAR(40) DEFAULT ''"),
        ("pbo", "NUMERIC(6,4) DEFAULT 0.0"),
        ("dsr", "NUMERIC(6,4) DEFAULT 0.0"),
        ("num_trials", "INT DEFAULT 0"),
    ]:
        try:
            if col == "lookback_days":
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {TABLE} RENAME COLUMN lookback_days TO lookback_days_old"))
                    conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN backtest_start_date VARCHAR(8)"))
                    conn.execute(text(f"UPDATE {TABLE} SET backtest_start_date = lookback_days_old::TEXT"))
                    conn.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN lookback_days_old"))
            else:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS {col} {typ}"))
        except Exception:
            pass


def get_last_run(engine: Any) -> dict[str, Any] | None:
    sql = text(f"""
        SELECT run_time, frequency, backtest_start_date, out_of_sample_days,
               initial_cash, params, sharpe, total_return, max_drawdown, status,
               config_hash, data_version, git_commit
        FROM {TABLE}
        ORDER BY run_time DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).mappings().fetchone()
    if row is None:
        return None
    result = dict(row)
    if isinstance(result.get("params"), str):
        result["params"] = json.loads(result["params"])
    return result


def should_rerun(
    last_run: dict[str, Any] | None,
    frequency: str,
    today: date | None = None,
    data_version: str | None = None,
    config_hash: str | None = None,
) -> tuple[bool, str]:
    """判断是否需要重新执行回测（四方绑定：数据版本 + 配置哈希 + 频率 + 时间）。

    data_version/config_hash 与上次成功记录不一致时，即使在同一周期内
    也强制重跑——上次结果对应的数据/配置已过期，缓存复用会静默失真。

    Returns:
        (should_run, reason)
    """
    if today is None:
        today = date.today()

    if last_run is None:
        return True, "从未执行过回测"

    last_time: datetime = last_run["run_time"]
    if isinstance(last_time, str):
        last_time = datetime.fromisoformat(last_time)
    last_date = last_time.date()

    # ── 四方绑定 1/2：数据版本 / 配置哈希 ──
    if data_version is not None:
        _last_dv = last_run.get("data_version") or ""
        if _last_dv and _last_dv != data_version:
            return True, f"数据版本变化（{_last_dv} → {data_version}），需重新回测"
    if config_hash is not None:
        _last_ch = last_run.get("config_hash") or ""
        if _last_ch and _last_ch != config_hash:
            return True, f"配置哈希变化（{_last_ch} → {config_hash}），需重新回测"

    # ── 上次失败保护：data/config 未变且上次失败，不再重复浪费算力 ──
    _last_status = last_run.get("status", "success")
    if _last_status == "failed":
        logger.warning(
            f"上次回测失败（于 {last_date}），数据/配置未发生变化，跳过重复执行。"
            f"如需强制重跑请调用 run_backtest_pipeline(force=True)"
        )
        return False, f"上次回测于 {last_date} 失败（data/config 未变），跳过重复执行"

    if frequency == "initial":
        return False, f"频率=initial，上次执行于 {last_date}，不再自动重跑"

    if frequency == "monthly":
        if last_date.year == today.year and last_date.month == today.month:
            return False, f"本月已于 {last_date} 执行过回测"
        return True, f"上月回测于 {last_date}，本月未执行"

    if frequency == "quarterly":
        last_q = (last_date.month - 1) // 3
        cur_q = (today.month - 1) // 3
        if last_date.year == today.year and last_q == cur_q:
            return False, f"本季度已于 {last_date} 执行过回测"
        return True, f"上季度回测于 {last_date}，本季度未执行"

    return True, f"未知频率 {frequency}，执行回测"


def _pyval(v: Any) -> Any:
    """numpy → 原生 Python 类型，避免 psycopg2 序列化成 np.float64(...) 导致 SQL 报错。

    P2 修复：兜底逻辑改用 str() 替代 try/except json.dumps，覆盖更多不可序列化类型
    （如 SQLAlchemy Engine、pandas DataFrame、自定义类等）。
    """
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: _pyval(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_pyval(item) for item in v)
    # FIX(P2) 兜底：json 不支持的类型 → 用 repr 而非 json.dumps 探测 + 直接 str(v) 转换
    # 旧逻辑：try/except json.dumps(v) 在某些 C 扩展类型（如 SQLAlchemy Engine）
    #   上可能抛出非 TypeError 异常（如 RuntimeError），导致未捕获崩溃。
    # 新逻辑：对一切非标量/非容器直接 str(v) 降级。
    if not isinstance(v, (str, int, float, bool, type(None))):
        return f"<{type(v).__module__}.{type(v).__qualname__}: {str(v)[:120]}>"
    return v


def record_run(
    engine: Any,
    frequency: str,
    backtest_start_date: str,
    out_of_sample_days: int,
    initial_cash: float,
    params: dict[str, float],
    sharpe: float,
    total_return: float,
    max_drawdown: float,
    status: str = "success",
    extra_metrics: dict[str, Any] | None = None,
    git_commit: str = "",
    config_hash: str = "",
    data_version: str = "",
) -> None:
    metrics = dict(extra_metrics or {})
    sortino = metrics.pop("sortino_ratio", None) or metrics.get("sortino", 0)
    calmar = metrics.pop("calmar_ratio", None) or metrics.get("calmar", 0)
    var_95 = metrics.pop("var_95", 0)
    cvar_95 = metrics.pop("cvar_95", 0)
    win_rate = metrics.pop("win_rate", 0)
    profit_factor = metrics.pop("profit_factor", 0)
    total_trades = metrics.pop("total_trades", 0)
    pbo = metrics.pop("pbo", 0.0)
    dsr = metrics.pop("dsr", 0.0)
    num_trials = metrics.pop("num_trials", 0)

    sql = text(f"""
        INSERT INTO {TABLE}
            (run_time, frequency, backtest_start_date, out_of_sample_days,
             initial_cash, params, sharpe, total_return, max_drawdown, status,
             sortino, calmar, var_95, cvar_95, win_rate, profit_factor, total_trades,
             git_commit, config_hash, data_version, pbo, dsr, num_trials)
        VALUES
            (NOW(), :frequency, :backtest_start_date, :out_of_sample_days,
             :initial_cash, CAST(:params AS jsonb), :sharpe, :total_return, :max_drawdown, :status,
             :sortino, :calmar, :var_95, :cvar_95, :win_rate, :profit_factor, :total_trades,
             :git_commit, :config_hash, :data_version, :pbo, :dsr, :num_trials)
    """)
    with engine.begin() as conn:
        conn.execute(sql, _pyval({
            "frequency": frequency,
            "backtest_start_date": backtest_start_date,
            "out_of_sample_days": out_of_sample_days,
            "initial_cash": initial_cash,
            "params": json.dumps(_pyval(params), ensure_ascii=False),
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "status": status,
            "sortino": sortino,
            "calmar": calmar,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "git_commit": git_commit,
            "config_hash": config_hash,
            "data_version": data_version,
            "pbo": pbo,
            "dsr": dsr,
            "num_trials": num_trials,
        }))
    logger.info(f"回测记录已写入 {TABLE}")


# ═══════════════════════════════════════════════════════════════
# 多重测试惩罚（Multiple Testing Deception）
# ═══════════════════════════════════════════════════════════════

def count_tuning_attempts(
    engine: Any,
    backtest_start_date: str,
    out_of_sample_days: int,
) -> int:
    """统计同区间（相同数据起始日期 + 相同 OOS 天数）的调参尝试次数。

    包含所有状态的记录（success/failure），因为即使失败也是用户的一次试错。
    调用时机决定口径：在 record_run 之前调用时需自行 +1（含本次）；
    在 record_run 之后调用则直接取返回值（已含本次）。

    Args:
        engine: SQLAlchemy engine。
        backtest_start_date: 回测起始日期（如 20230101）。
        out_of_sample_days: 样本外天数。

    Returns:
        累计调参次数。
    """
    sql = text(f"""
        SELECT COUNT(*)
        FROM {TABLE}
        WHERE backtest_start_date = :bstart
          AND out_of_sample_days = :oos_days
    """)
    with engine.connect() as conn:
        count = conn.execute(sql, {"bstart": backtest_start_date, "oos_days": out_of_sample_days}).scalar()
    return int(count) if count else 0


def apply_multiple_testing_penalty(
    sharpe: float,
    sortino: float,
    attempt_count: int,
    backtest_start_date: str,
    out_of_sample_days: int,
) -> tuple[float, float, str]:
    """对高频调参施加统计学惩罚。

    业务规则：
    - attempt_count <= MAX_TUNING_ATTEMPTS: 无惩罚
    - attempt_count > MAX_TUNING_ATTEMPTS: Sharpe 和 Sortino 各硬扣 20%
    - attempt_count > 30: 标记 CRITICAL

    Args:
        sharpe: 原始 Sharpe。
        sortino: 原始 Sortino。
        attempt_count: 同区间累计调参次数（含本次）。
        backtest_start_date: 回测起始日期。
        out_of_sample_days: 样本外天数。

    Returns:
        (punished_sharpe, punished_sortino, warning_level)
        warning_level 为 INFO / WARNING / CRITICAL
    """
    period_key = f"{backtest_start_date}/OOS{out_of_sample_days}"

    if attempt_count <= MAX_TUNING_ATTEMPTS:
        logger.info(
            f"[多重测试] 同区间调参 {attempt_count}/{MAX_TUNING_ATTEMPTS} 次，未超限，无需惩罚"
        )
        return sharpe, sortino, "INFO"

    # ── 超限惩罚 ──
    punished_sharpe = sharpe * (1.0 - MULTIPLE_TESTING_PENALTY)
    punished_sortino = sortino * (1.0 - MULTIPLE_TESTING_PENALTY)

    warning_level = "WARNING"
    if attempt_count > 30:
        warning_level = "CRITICAL"
        logger.critical(
            f"[多重测试惩罚] 🔴 CRITICAL：同区间 {period_key} 已调参 {attempt_count} 次！"
            f" Sharpe {sharpe:.4f} → {punished_sharpe:.4f}（扣减 {MULTIPLE_TESTING_PENALTY:.0%}）"
            f" 该策略存在严重过拟合风险，建议重新审视特征工程和参数空间设计"
        )

    logger.warning(
        f"[多重测试惩罚] ⚠️ 高危：同区间 {period_key} 已调参 {attempt_count} 次（上限 {MAX_TUNING_ATTEMPTS}），"
        f"触发统计学惩罚 Sharpe {sharpe:.4f} → {punished_sharpe:.4f}（扣减 {MULTIPLE_TESTING_PENALTY:.0%}）"
        f" Sortino {sortino:.4f} → {punished_sortino:.4f}"
    )

    return punished_sharpe, punished_sortino, warning_level
