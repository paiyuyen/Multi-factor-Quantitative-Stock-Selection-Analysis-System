"""Deep log analysis: key events and patterns."""
import re
from collections import Counter

LOG_PATH = r"C:\Users\y84189905\Downloads\CoreNews_Reports\Logs\backtest_20260825_014315.log"

# --- Phase 1: Extract structured timeline ---
phases = []
with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, 1):
        # Extract timestamp
        m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        ts = m.group(1) if m else ""
        
        # Categorize line
        if "回测管线" in line:
            phases.append((i, ts, "PIPELINE", line.strip()[:200]))
        elif "数据同步" in line or "IncrementalSync" in line or "batch" in line or "同步" in line:
            phases.append((i, ts, "SYNC", line.strip()[:200]))
        elif "prepare" in line.lower() or "indicator" in line.lower() or "finalize" in line:
            phases.append((i, ts, "PREPARE", line.strip()[:200]))
        elif "bayesian" in line.lower() or "WFO" in line or "Sobol" in line or "Bayes" in line:
            if "ERROR" not in line and "WARNING" not in line and "CRITICAL" in line or True:
                if len(phases) == 0 or phases[-1][2] != "WFO":
                    phases.append((i, ts, "WFO", line.strip()[:200]))
                elif phases[-1][2] == "WFO" and phases[-1][0] > i - 5:
                    phases.append((i, ts, "WFO", line.strip()[:200]))
        elif "CRITICAL" in line or "ERROR" in line:
            phases.append((i, ts, "ERR", line.strip()[:250]))
        elif "回测完成" in line or "校准完成" in line or "pipeline" in line.lower():
            phases.append((i, ts, "RESULT", line.strip()[:250]))
        elif "config.ini" in line or "采纳门控" in line:
            phases.append((i, ts, "GATE", line.strip()[:250]))
        elif "prepare" in line and ("ML" in line or "XGBoost" in line or "信号覆写" in line):
            phases.append((i, ts, "ML", line.strip()[:200]))
        elif "DataSync" in line or "数据同步" in line or "stock_daily_kline" in line:
            phases.append((i, ts, "DATA", line.strip()[:200]))

print("=" * 70)
print("PHASE TIMELINE")
print("=" * 70)
cur_phase = None
for line_num, ts, phase, text in phases:
    if phase != cur_phase:
        cur_phase = phase
        if cur_phase not in ("WFO",):  # skip wfo spam
            print(f"\n  --- {cur_phase} ---")
    # Only print WFO phase changes
    if phase in ("PIPELINE", "SYNC", "PREPARE", "ERR", "RESULT", "GATE", "ML", "DATA"):
        print(f"    {ts} | {text}")
    elif phase == "WFO" and ("CRITICAL" in text or "完成" in text or "窗口" in text or "WFO" in text.split("|")[-1][:50]):
        print(f"    {ts} | {text}")

print()
print("=" * 70)
print("TIMING ANALYSIS: Key duration measurements")
print("=" * 70)

# Extract timestamps for key events
import datetime
keys = {}
with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        ts_m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if not ts_m:
            continue
        ts = datetime.datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")
        
        if "开始回测管线" in line and ts not in keys:
            keys["pipeline_start"] = ts
        elif "数据同步" in line and "同步" in line and "开始" in line and ts not in keys.get("data_start", {}):
            if "data_start" not in keys:
                keys["data_start"] = ts
        elif "回测完成" in line or "Sharpe=-0.85" in line:
            if "backtest_end" not in keys:
                keys["backtest_end"] = ts
        elif "OOS衰减校验.*PASS" in line.replace(" ", ""):
            if "oos_pass" not in keys:
                keys["oos_pass"] = ts
        elif "OOS衰减校验.*FAIL" in line.replace(" ", ""):
            if "oos_fail" not in keys:
                keys["oos_fail"] = ts
        elif "采纳门控" in line and "不予保存" in line:
            if "gate_reject" not in keys:
                keys["gate_reject"] = ts
        elif "回测校准完成" in line:
            keys["final"] = ts
        elif "XGBoost 重训" in line:
            if "xgboost_runs" not in keys:
                keys["xgboost_runs"] = []
            keys["xgboost_runs"].append(ts)

for k, v in sorted(keys.items(), key=lambda x: str(x[1]) if not isinstance(x[1], list) else "0"):
    print(f"  {k}: {v}")

# Compute durations
if "pipeline_start" in keys and "final" in keys:
    total = keys["final"] - keys["pipeline_start"]
    print(f"\n  Total pipeline duration: {total} ({total.total_seconds()/3600:.1f} hours)")

# Count XGBoost retrain runs
if "xgboost_runs" in keys:
    runs = keys["xgboost_runs"]
    print(f"\n  XGBoost retrain runs: {len(runs)}")
    if len(runs) > 1:
        # Time between runs
        for j in range(1, len(runs)):
            d = runs[j] - runs[j-1]
            print(f"    Run {j}→{j+1}: {d}")

print()
print("=" * 70)
print("SSL ERROR TIMING (when do they occur?)")
print("=" * 70)
ssl_times = []
with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "SSLError" in line:
            ts_m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if ts_m:
                ssl_times.append(ts_m.group(1))

if ssl_times:
    first = ssl_times[0]
    last = ssl_times[-1]
    unique_stocks = set()
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "SSLError" in line:
                m = re.search(r'(sh\d{6}|sz\d{6})', line)
                if m:
                    unique_stocks.add(m.group(1))
    print(f"  First SSL error: {first}")
    print(f"  Last SSL error:  {last}")
    print(f"  Total SSL errors: {len(ssl_times)}")
    print(f"  Unique stocks affected: {len(unique_stocks)}")
    print(f"  Year 2026 data affected (future dates → no real data)")

print()
print("=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)
# Look for data warnings
data_warnings = []
with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if "WARNING" in line and ("data" in line.lower() or "缺失" in line or "空" in line or 
                                    "停牌" in line or "复权" in line or "质量" in line):
            data_warnings.append(line.strip()[:200])

# Unique warnings
seen = set()
for w in data_warnings:
    # Deduplicate by first 150 chars
    key = w[:150]
    if key not in seen:
        seen.add(key)
        print(f"  {w}")
        if len(seen) >= 30:
            break

print()
print(f"  Total unique data warnings: ~{len(seen)}")
