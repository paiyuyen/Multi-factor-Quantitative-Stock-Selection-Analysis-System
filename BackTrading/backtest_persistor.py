"""Task P2.3: 回测结果持久化接入 DB。

将回测 run 元数据、权益曲线和交易流水写入 PostgreSQL，支持：
- `backtest_runs` 主表：run_id, start_date, end_date, sharpe, max_dd, config_hash 等
- `backtest_equity` 明细表：按 run_id 关联的净值日序列
- `backtest_trades` 明细表：按 run_id 关联的交易流水
- 幂等写入：同 run_id 的 upsert（DELETE + INSERT）

调用方在 `runner.py` / `simulated_trading.py` 回测入口完成后调用
`persist_run(run_id, params, trade_log, equity_curve, metrics)` 完成全量落盘。
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

# ── 表结构（DDL 首次运行时自动建表）─────────────────────

DDL_TABLES = {
    "backtest_runs": """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id          VARCHAR(64)  NOT NULL PRIMARY KEY,
            strategy_name   VARCHAR(128),
            start_date      VARCHAR(10),
            end_date        VARCHAR(10),
            total_return    NUMERIC,
            sharpe_ratio    NUMERIC,
            max_drawdown    NUMERIC,
            win_rate        NUMERIC,
            num_trades      INTEGER,
            config_hash     VARCHAR(64),
            params_json     TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "backtest_equity": """
        CREATE TABLE IF NOT EXISTS backtest_equity (
            run_id    VARCHAR(64)  NOT NULL,
            trade_date VARCHAR(10) NOT NULL,
            portfolio_value NUMERIC,
            cash        NUMERIC,
            position_value NUMERIC,
            turnover    NUMERIC,
            PRIMARY KEY (run_id, trade_date)
        )
    """,
    "backtest_trades": """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            run_id    VARCHAR(64)   NOT NULL,
            seq       BIGINT        NOT NULL,
            trade_date VARCHAR(10),
            symbol    VARCHAR(16),
            action    VARCHAR(16),
            price     NUMERIC,
            quantity  BIGINT,
            value     NUMERIC,
            cost      NUMERIC,
            PRIMARY KEY (run_id, seq)
        )
    """,
}


class BacktestPersistor:
    """回测结果持久化器。

    将回测结果写入 PostgreSQL。若数据库不可用则静默跳过（不回退报错）。
    """

    def __init__(self, engine=None) -> None:
        self.engine = engine
        self._tables_created = False

    def _get_engine(self):
        """惰性获取数据库引擎。"""
        if self.engine is None:
            try:
                from DataManager.DbEngine import get_engine as _get_engine
                from UtilsManager.ConfigParser import Config
                self.engine = _get_engine(Config())
            except Exception as e:
                logger.warning(f"[P2.3] 数据库引擎不可用，回测持久化跳过: {e}")
                self.engine = None
        return self.engine

    def ensure_tables(self) -> bool:
        """建表（幂等：CREATE IF NOT EXISTS）。返回是否成功。"""
        if self._tables_created:
            return True
        eng = self._get_engine()
        if eng is None:
            return False
        try:
            with eng.begin() as conn:
                for ddl in DDL_TABLES.values():
                    conn.execute(text(ddl))
            self._tables_created = True
            logger.info("[P2.3] backtest_runs / backtest_equity / backtest_trades 表就绪")
            return True
        except Exception as e:
            logger.warning(f"[P2.3] 建表失败，持久化降级: {e}")
            return False

    # ── 核心入口 ─────────────────────────────────────

    def persist_run(
        self,
        *,
        run_id: str,
        params: dict[str, Any] | None = None,
        trade_log: list[dict[str, Any]] | pd.DataFrame | None = None,
        equity_curve: list[dict[str, Any]] | pd.DataFrame | None = None,
        metrics: dict[str, Any] | None = None,
        strategy_name: str = "unknown",
    ) -> bool:
        """将单次回测结果落盘。

        Args:
            run_id: 回测唯一标识（如 "2026-08-26_MOMENTUM_v1"）。
            params: 参数字典（写入 params_json 并计算 config_hash）。
            trade_log: 交易流水（list[dict] 或 DataFrame）。
            equity_curve: 净值曲线（list[dict] 或 DataFrame）。
            metrics: 绩效指标字典。
            strategy_name: 策略名称。

        Returns:
            是否成功写入 DB。
        """
        if not self.ensure_tables():
            return False

        # ── 计算 config_hash ──
        cfg_hash = _hash_config(params or {})

        # ── 提取 start_date / end_date / num_trades ──
        start_date, end_date, num_trades = _extract_dates(equity_curve, trade_log, params)

        # ── 写入/更新 backtest_runs 主表 ──
        self._upsert_run(
            run_id=run_id,
            strategy_name=str(strategy_name),
            start_date=start_date,
            end_date=end_date,
            metrics=metrics or {},
            cfg_hash=cfg_hash,
            params_json=json.dumps(params or {}, ensure_ascii=False, default=str),
            num_trades=num_trades,
        )

        # ── 写入明细：权益曲线 & 交易流水 ──
        if equity_curve is not None:
            self._write_equity(run_id, equity_curve)
        if trade_log is not None:
            self._write_trades(run_id, trade_log)

        logger.info(f"[P2.3] 回测结果持久化完成 run_id={run_id}")
        return True

    # ── 内部写入方法 ──────────────────────────────────

    @staticmethod
    def _sanitize_for_sql(val: Any) -> Any:
        """清理 NaN/Inf，避免 PostgreSQL 拒绝。"""
        if val is None:
            return None
        if isinstance(val, float) and (not np.isfinite(val)):
            return None
        if isinstance(val, np.floating):
            f = float(val)
            return None if not np.isfinite(f) else f
        if isinstance(val, (np.integer,)):
            return int(val)
        return val

    def _upsert_run(
        self,
        run_id: str,
        strategy_name: str,
        start_date: str | None,
        end_date: str | None,
        metrics: dict[str, Any],
        cfg_hash: str,
        params_json: str,
        num_trades: int,
    ) -> None:
        """幂等写入 backtest_runs（先 DELETE 后 INSERT）。"""
        eng = self._get_engine()
        if eng is None:
            return
        try:
            with eng.begin() as conn:
                # 删除旧记录再插入（简单幂等方案）
                conn.execute(text("DELETE FROM backtest_runs WHERE run_id = :rid"), {"rid": run_id})
                conn.execute(
                    text("""
                        INSERT INTO backtest_runs
                            (run_id, strategy_name, start_date, end_date,
                             total_return, sharpe_ratio, max_drawdown, win_rate,
                             num_trades, config_hash, params_json, updated_at)
                        VALUES (:run_id, :strategy_name, :start_date, :end_date,
                                :total_return, :sharpe, :max_dd, :win_rate,
                                :num_trades, :config_hash, :params_json, CURRENT_TIMESTAMP)
                    """),
                    {
                        "run_id": run_id,
                        "strategy_name": strategy_name,
                        "start_date": start_date,
                        "end_date": end_date,
                        "total_return": self._sanitize_for_sql(metrics.get("total_return")),
                        "sharpe": self._sanitize_for_sql(metrics.get("sharpe_ratio")),
                        "max_dd": self._sanitize_for_sql(metrics.get("max_drawdown")),
                        "win_rate": self._sanitize_for_sql(metrics.get("win_rate")),
                        "num_trades": num_trades,
                        "config_hash": cfg_hash,
                        "params_json": params_json,
                    },
                )
        except Exception as e:
            logger.warning(f"[P2.3] backtest_runs 写入失败 run_id={run_id}: {e}")

    def _write_equity(self, run_id: str, equity: Any) -> None:
        """写入 backtest_equity 明细。"""
        eng = self._get_engine()
        if eng is None:
            return
        try:
            df = _equity_to_df(equity)
            if df.empty:
                return
            # 清理 run_id
            with eng.begin() as conn:
                conn.execute(text("DELETE FROM backtest_equity WHERE run_id = :rid"), {"rid": run_id})
                rows = []
                for _, r in df.iterrows():
                    rows.append({
                        "run_id": run_id,
                        "trade_date": str(r.get("time", r.get("trade_date", "")))[:10],
                        "portfolio_value": self._sanitize_for_sql(r.get("portfolio_value")),
                        "cash": self._sanitize_for_sql(r.get("cash")),
                        "position_value": self._sanitize_for_sql(r.get("position_value")),
                        "turnover": self._sanitize_for_sql(r.get("turnover")),
                    })
                if rows:
                    conn.execute(text("INSERT INTO backtest_equity VALUES "
                                       "(:run_id, :trade_date, :portfolio_value, "
                                       ":cash, :position_value, :turnover)"),
                                 rows)
            logger.debug(f"[P2.3] 写入权益曲线 run_id={run_id} rows={len(rows)}")
        except Exception as e:
            logger.warning(f"[P2.3] 权益曲线写入失败 run_id={run_id}: {e}")

    def _write_trades(self, run_id: str, trade_log: Any) -> None:
        """写入 backtest_trades 明细。"""
        eng = self._get_engine()
        if eng is None:
            return
        try:
            df = _trades_to_df(trade_log)
            if df.empty:
                return
            with eng.begin() as conn:
                conn.execute(text("DELETE FROM backtest_trades WHERE run_id = :rid"), {"rid": run_id})
                rows = []
                for seq, r in df.iterrows():
                    rows.append({
                        "run_id": run_id,
                        "seq": int(seq),
                        "trade_date": str(r.get("time", ""))[:10],
                        "symbol": str(r.get("symbol", "")),
                        "action": str(r.get("action", "")),
                        "price": self._sanitize_for_sql(r.get("price")),
                        "quantity": int(r.get("qty", 0)),
                        "value": self._sanitize_for_sql(r.get("value")),
                        "cost": self._sanitize_for_sql(r.get("cost")),
                    })
                if rows:
                    conn.execute(text("INSERT INTO backtest_trades VALUES "
                                       "(:run_id, :seq, :trade_date, :symbol, "
                                       ":action, :price, :quantity, :value, :cost)"),
                                 rows)
            logger.debug(f"[P2.3] 写入交易流水 run_id={run_id} rows={len(rows)}")
        except Exception as e:
            logger.warning(f"[P2.3] 交易流水写入失败 run_id={run_id}: {e}")


# ── 工具函数 ──────────────────────────────────────────────

def _hash_config(params: dict[str, Any]) -> str:
    """计算参数配置哈希（64 位 hex）。"""
    raw = json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def _extract_dates(
    equity: Any, trade_log: Any, params: dict
) -> tuple[str | None, str | None, int]:
    """从 equity 和 trade_log 中提取 start_date, end_date, num_trades。"""
    start_date = None
    end_date = None
    num_trades = 0

    if equity is not None:
        df = _equity_to_df(equity)
        dates = []
        for col in ("time", "trade_date"):
            if col in df.columns:
                dates = df[col].dropna().astype(str).str[:10].tolist()
                break
        if dates:
            start_date = dates[0]
            end_date = dates[-1]

    if trade_log is not None:
        df = _trades_to_df(trade_log)
        num_trades = len(df)
        if not start_date and "time" in df.columns:
            ts = df["time"].dropna().astype(str).str[:10].tolist()
            if ts:
                start_date = ts[0]
                end_date = ts[-1]

    return start_date, end_date, num_trades


def _equity_to_df(equity: Any) -> pd.DataFrame:
    """兼容 list[dict] / DataFrame → DataFrame。"""
    if isinstance(equity, list):
        return pd.DataFrame(equity) if equity else pd.DataFrame()
    if isinstance(equity, pd.DataFrame):
        return equity.copy()
    return pd.DataFrame()


def _trades_to_df(trade_log: Any) -> pd.DataFrame:
    """兼容 list[dict] / DataFrame → DataFrame。"""
    if isinstance(trade_log, list):
        return pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    if isinstance(trade_log, pd.DataFrame):
        return trade_log.copy()
    return pd.DataFrame()
