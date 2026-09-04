#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单只股票技术指标分析工具

让用户输入一个股票代码，从 akshare 下载数据并复用现有分析类，
输出 Excel 报告中所有的技术指标因子结论。

用法：
    python TreasureBox/SingleStockAnalyzer.py          # 交互模式
    python TreasureBox/SingleStockAnalyzer.py 000001   # 命令行参数模式

依赖：akshare, pandas, pandas-ta
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台 UTF-8 输出
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import traceback

import pandas as pd
import requests

# ── 从项目现有模块导入 ─────────────────────────────────────────────────────
from UtilsManager.ConfigParser import Config
from UtilsManager.CodeNormalizer import CodeNormalizer

# ── 打印辅助函数 ───────────────────────────────────────────────────────────
WIDTH = 68


def print_header() -> None:
    print()
    print("=" * WIDTH)
    print("  单只股票技术指标分析工具")
    print("=" * WIDTH)


def print_section(title: str) -> None:
    print()
    print("-" * WIDTH)
    print(f"  {title}")
    print("-" * WIDTH)


def print_field(label: str, value: Any) -> None:  # noqa: ANN401
    if value is not None and str(value).strip():
        print(f"    {label:<26} : {value}")


# ── 腾讯行情 API ─────────────────────────────────────────────────────────
TENCENT_KLINE_URL = "http://ifzq.gtimg.cn/appstock/app/fqkline/get"


def _fetch_stock_name(symbol: str) -> str:
    """通过腾讯行情接口获取股票简称"""
    try:
        r = requests.get(f"http://qt.gtimg.cn/q={symbol}", timeout=10)
        txt = r.text
        if "~" in txt:
            parts = txt.split("~")
            name = parts[1] if len(parts) > 1 else symbol
        else:
            name = symbol
    except Exception:
        name = symbol
    return name


def _tencent_kline(symbol: str, days: int) -> pd.DataFrame | None:
    """
    直接调用腾讯行情 API 获取后复权日 K 线。
    API 返回 [date, open, close, high, low, volume]（成交量单位为股）。
    """
    params = {"param": f"{symbol},day,,,{days},hfq"}
    try:
        r = requests.get(TENCENT_KLINE_URL, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  [ERROR] 腾讯 API 请求失败: {e}")
        return None

    if data.get("code") != 0:
        print(f"  [ERROR] 腾讯 API 返回错误: {data.get('msg')}")
        return None

    records = data.get("data", {}).get(symbol, {}).get("hfqday")
    if not records:
        print(f"  [ERROR] 未获取到 K 线数据 (symbol={symbol})")
        return None

    rows = []
    for row in records:
        if len(row) < 6:
            continue
        try:
            rows.append({
                "date": pd.to_datetime(row[0]),
                "open": float(row[1]),
                "close": float(row[2]),
                "high": float(row[3]),
                "low": float(row[4]),
                "volume": float(row[5]),
            })
        except (ValueError, TypeError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_kline_data(symbol: str, days: int = 300) -> pd.DataFrame | None:
    """
    获取个股后复权 K 线数据，返回 OHLCV 日线（重试 3 次）。

    Returns:
        DataFrame | None: 包含 date/open/close/high/low/volume 的 DataFrame
    """
    for attempt in range(3):
        df = _tencent_kline(symbol, days)
        if df is not None and not df.empty:
            return df
        if attempt < 2:
            wait = 2 ** attempt
            print(f"  [RETRY] 第 {attempt + 1} 次失败，{wait} 秒后重试...")
            time.sleep(wait)
    return None


# ── 单只股票分析 ──────────────────────────────────────────────────────────
def analyze_stock(raw_code: str) -> None:
    pure_code = CodeNormalizer.normalize(raw_code)
    symbol = CodeNormalizer.add_market_prefix(raw_code)
    print(f"\n  [-] 股票代码: {pure_code}  ({symbol})")

    # 2. 获取 K 线数据 ────────────────────────────────────────────────
    print(f"\n  >>> 正在从 akshare 下载数据 ({pure_code})...")
    df = fetch_kline_data(symbol, days=300)
    if df is None or df.empty or len(df) < 30:
        print("  [ERROR] 数据不足（至少需要 30 个交易日）")
        return

    print(f"  [OK] 获取到 {len(df)} 条日 K 线数据")
    print(f"      日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}")

    # 3. 基础信息 ──────────────────────────────────────────────────────
    print_section("基础信息")
    print_field("股票代码", pure_code)

    # 尝试获取股票名称（腾讯行情接口）
    stock_name = _fetch_stock_name(symbol)
    print_field("股票名称", stock_name)

    latest_price = df["close"].iloc[-1]
    print_field("最新价", f"{latest_price:.2f}")
    print_field("数据条数", len(df))

    # 4. 复用 TASignalProcessor 计算全部技术信号 ──────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = Config(config_file=os.path.join(project_root, "config.ini"))

    # 准备 TASignalProcessor 要求的 hist_df 格式
    hist_df = df.copy()
    hist_df["股票代码"] = pure_code

    from LogicAnalyzer.SignalManager import TASignalProcessor

    processor = TASignalProcessor(None, config=config)
    result = processor._process_single_stock(symbol, hist_df)

    if result is None:
        print("  [ERROR] 技术指标分析失败")
        return

    # 5. MACD 指标 ────────────────────────────────────────────────────
    print_section("MACD 指标")
    print_field("MACD信号", result.get("macd_signal", ""))
    print_field("MACD趋势分类", result.get("pipeline", {}).get("macd_trend", ""))

    # 6. MACD趋势评分 ────────────────────────────────────────────────
    print_section("MACD 趋势评分")
    bull_result = result.get("pipeline")
    if bull_result:
        details = bull_result.get("details", {})
        if details:
            print()
            print("  " + "\u2500" * 18)
            for dim_key in ["MACD趋势", "金叉信号", "柱状动能", "DIF斜率", "背离信号", "量价配合", "K线形态"]:
                dim_val = details.get(dim_key, {})
                desc = dim_val.get("desc", "")
                score = dim_val.get("score", 0)
                print(f"    {dim_key:<20} : {score:>3}  ({desc})")

    # 7. KDJ / CCI / RSI / BOLL ──────────────────────────────────────
    print_section("KDJ 指标")
    print_field("KDJ_Signal", result.get("kdj_signal", "无信号"))

    print_section("CCI 指标")
    print_field("CCI_Signal", result.get("cci_signal", "无信号"))

    print_section("RSI 指标")
    print_field("RSI_Signal", result.get("rsi_signal", "无信号"))

    print_section("BOLL 指标")
    print_field("BOLL_Signal", result.get("boll_signal", "无信号"))

    # 8. 汇总 ────────────────────────────────────────────────────────
    print()
    print("=" * WIDTH)
    if bull_result:
        score = bull_result.get("score", 0)
        if score >= 80:
            rating = "[强烈买入]"
        elif score >= 60:
            rating = "[逢低布局]"
        elif score >= 40:
            rating = "[观望为主]"
        else:
            rating = "[回避/做空]"
        print(f"  综合评分: {score}  {rating}")
        print(f"  综合结论: {bull_result.get('conclusion', 'N/A')}")
    print("=" * WIDTH)
    print()


# ── 主流程 ────────────────────────────────────────────────────────────────
def main() -> None:
    print_header()

    if len(sys.argv) > 1:
        first_code = sys.argv[1]
        analyze_stock(first_code)
        if first_code:
            _ask_continue()
        return

    while True:
        raw_code = input("\n  请输入股票代码 (6位数字，如 000001，输入 quit 退出): ").strip()
        if raw_code.lower() in ("quit", "q"):
            print("  再见！")
            break
        if not raw_code.isdigit() or len(raw_code) != 6:
            print(f"  [ERROR] 输入不合法: {raw_code!r} (需为 6 位数字或 quit)")
            continue
        analyze_stock(raw_code)


if __name__ == "__main__":
    main()
