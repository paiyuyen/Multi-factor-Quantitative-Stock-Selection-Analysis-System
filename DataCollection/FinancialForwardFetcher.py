from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class FinancialForwardFetcher:
    """财务前瞻因子获取器 — 业绩预告超预期 + 分析师一致预期调整。

    用 akshare 获取：
      - stock_profit_forecast_em    → 业绩预告类型 + 净利润变动区间
      - stock_analyst_rank_em       → 分析师评级分布（买入/增持/中性/减持）
    """

    CACHE_DIR: str | None = None

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, "TEMP_DATA_DIRECTORY"):
            self.CACHE_DIR = config.TEMP_DATA_DIRECTORY
        else:
            self.CACHE_DIR = os.path.expanduser("~/Downloads/CoreNews_Reports/cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def fetch_forecasts(self) -> pd.DataFrame:
        """获取全市场业绩预告。

        Returns:
            DataFrame with columns: symbol, 预告类型, 净利润变动_min, 净利润变动_max
        """
        cache_path = os.path.join(self.CACHE_DIR, "financial_forecast.csv")
        if os.path.exists(cache_path):
            modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - modified).days < 1:
                cached = pd.read_csv(cache_path, dtype={"symbol": str})
                logger.info(f"[财务前瞻] 业绩预告缓存命中 ({len(cached)} 条)")
                return cached

        try:
            import akshare as ak
            df = ak.stock_profit_forecast_em()
        except Exception as e:
            logger.warning(f"[财务前瞻] akshare 业绩预告获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df.empty:
            return df

        # 解析 akshare 列名
        code_col = [c for c in df.columns if "代码" in c or "code" in c.lower()]
        type_col = [c for c in df.columns if "预告" in c or "type" in c.lower()]
        pmin_col = [c for c in df.columns if "下限" in c or "min" in c.lower() or "p_change_min" in c]
        pmax_col = [c for c in df.columns if "上限" in c or "max" in c.lower() or "p_change_max" in c]

        result = pd.DataFrame()
        if code_col:
            result["symbol"] = df[code_col[0]].astype(str).str.strip().str.zfill(6)
        else:
            return pd.DataFrame()

        if type_col:
            result["预告类型"] = df[type_col[0]].astype(str)
        else:
            result["预告类型"] = ""

        # 超预期程度 = (min + max) / 2，正值=预增/扭亏
        if pmin_col and pmax_col:
            _min = pd.to_numeric(df[pmin_col[0]], errors="coerce").fillna(0)
            _max = pd.to_numeric(df[pmax_col[0]], errors="coerce").fillna(0)
            result["净利润变动_pct"] = (_min + _max) / 2
        else:
            result["净利润变动_pct"] = 0.0

        # 超预期分数：预增/扭亏/略增 → 正分；预减/首亏/略减 → 负分
        surprise_map = {
            "预增": 1.0, "扭亏": 1.0, "略增": 0.5, "续盈": 0.3,
            "续亏": -0.5, "略减": -0.5, "预减": -1.0, "首亏": -1.0,
            "不确定": 0.0,
        }
        result["业绩超预期分"] = result["预告类型"].map(surprise_map).fillna(0.0)

        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[财务前瞻] 业绩预告获取完成 ({len(result)} 条)")
        return result

    def fetch_analyst_ranks(self) -> pd.DataFrame:
        """获取全市场分析师评级分布（子进程隔离，防 Windows segfault）。

        Returns:
            DataFrame with columns: symbol, 买入占比, 增持占比, 中性占比, 减持占比
        """
        cache_path = os.path.join(self.CACHE_DIR, "analyst_rank.csv")
        if os.path.exists(cache_path):
            modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - modified).days < 1:
                cached = pd.read_csv(cache_path, dtype={"symbol": str})
                logger.info(f"[财务前瞻] 分析师评级缓存命中 ({len(cached)} 只)")
                return cached

        tmp_path = cache_path.replace(".csv", "_tmp.csv")
        script = (
            "import sys, os\n"
            "os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'\n"
            "import pandas as pd\n"
            "pd.options.future.infer_string = False\n"
            "pd.options.mode.string_storage = 'python'\n"
            "import akshare as ak\n"
            f"df = ak.stock_analyst_rank_em()\n"
            f"df.to_csv(r'{tmp_path}', index=False, encoding='utf-8-sig')\n"
            "print('OK')\n"
        )
        try:
            import subprocess
            r = subprocess.run(
                [sys.executable, "-c", script],
                timeout=60,
                capture_output=True,
                encoding="utf-8",
                errors="replace",  # Windows 子进程 stderr 可能输出 GBK，用 replace 避免 UnicodeDecodeError
            )
            if r.returncode != 0 or "OK" not in r.stdout:
                raise RuntimeError(f"subprocess failed (rc={r.returncode}): {r.stderr[:200]}")
            df = pd.read_csv(tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.warning(f"[财务前瞻] 分析师评级获取失败: {e}")
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, dtype={"symbol": str})
            return pd.DataFrame()

        if df.empty:
            return df

        code_col = [c for c in df.columns if "代码" in c or "code" in c.lower()]
        result = pd.DataFrame()
        if code_col:
            result["symbol"] = df[code_col[0]].astype(str).str.strip().str.zfill(6)
        else:
            return pd.DataFrame()

        # 找买入/增持/中性/减持列
        buy_col = [c for c in df.columns if "买入" in c]
        add_col = [c for c in df.columns if "增持" in c]
        hold_col = [c for c in df.columns if "中性" in c or "持有" in c]
        sell_col = [c for c in df.columns if "减持" in c or "卖出" in c]

        result["买入占比"] = pd.to_numeric(df[buy_col[0]], errors="coerce").fillna(0) if buy_col else 0.0
        result["增持占比"] = pd.to_numeric(df[add_col[0]], errors="coerce").fillna(0) if add_col else 0.0
        result["中性占比"] = pd.to_numeric(df[hold_col[0]], errors="coerce").fillna(0) if hold_col else 0.0
        result["减持占比"] = pd.to_numeric(df[sell_col[0]], errors="coerce").fillna(0) if sell_col else 0.0

        # 综合分析师共识分 = 买入×1 + 增持×0.5 - 减持×0.5
        result["分析师共识分"] = (
            result["买入占比"] * 1.0 + result["增持占比"] * 0.5 - result["减持占比"] * 0.5
        )

        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info(f"[财务前瞻] 分析师评级获取完成 ({len(result)} 只)")
        return result
