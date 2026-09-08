"""Phase 0: 全局指标预计算缓存。

在贝叶斯寻优开始前，为所有股票一次性预计算技术指标 + peak/trough。
后续每次 evaluation 直接加载缓存，只跑评分层 compute_signals。

缓存存储位置: CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet
                  + .peaks.npy + .troughs.npy + .meta.json

内存缓存仅在主进程有效；子进程（ProcessPoolExecutor worker）从磁盘加载。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading import output_store as _os
from BackTrading.vectorized_signal import (
    compute_param_independent_features as _p0_features,
    _p0_feature_constants as _p0_constants,
)

_IN_MEMORY: dict[str, pd.DataFrame] = {}
_PEAKS: dict[str, np.ndarray] = {}
_TROUGHS: dict[str, np.ndarray] = {}
_PRECOMPUTE_DONE: bool = False
# 背离检测缓存：div_type/div_idx/div_strength 只依赖股票自身 DIF 数据，与参数无关，
# 在 Phase 0 一次性预计算，后续每次 evaluation（不同参数组合）直接复用，
# 避免每轮贝叶斯迭代都重跑 O(n²) 的逐 bar Python 循环。
_DIVERGENCE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

# ── 批次隔离：WFO 多窗口/多路径下，不同数据切片的指标必须隔离 ──
# 2026-08-07 事故根因：窗口 [1-0] IS 的指标(2023-01-03~2025-11-10)常驻 _IN_MEMORY，
# 后续 OOS 验证的 worker 命中旧内存缓存，用 IS 日期算信号 → OOS 段信号全 NaN → 0 交易。
# _ACTIVE_FINGERPRINT = 当前已载入内存缓存的整批数据指纹（prepare 的 data_fp）；
# _SYMBOL_FPS = {symbol: 该股载入时的数据指纹}，供内存命中 O(1) 校验。
_ACTIVE_FINGERPRINT: str | None = None
_SYMBOL_FPS: dict[str, str] = {}


def _reset_memory_caches() -> None:
    """清空全部内存缓存（跨数据批次切换时调用，防指标污染）。"""
    _IN_MEMORY.clear()
    _PEAKS.clear()
    _TROUGHS.clear()
    _DIVERGENCE.clear()
    _SYMBOL_FPS.clear()


def _data_fingerprint(df: pd.DataFrame) -> str:
    """P1-2 修复：内容级强哈希（全列采样 sha256），替代首/中/尾3行+sum弱指纹。

    弱指纹问题：复权改写/因子修正只改中间行数据，首/中/尾3行+sum不敏感→
    静默复用脏指标缓存，信号全错但无告警。

    新方案：均匀步长采样所有 OHLCV 字节级 sha256 + 行数 + 日期范围 + adj_factor，
    内容变化即整库失效（宁可全废，不可错用）。与 prepare._data_fingerprint 口径对齐。
    """
    try:
        key_cols = ["close", "high", "low", "open", "volume"]
        present = [c for c in key_cols if c in df.columns]
        if not present:
            return "unknown"
        # 均匀采样（避免全量哈希性能问题），步长按总行数动态
        step = max(1, len(df) // 20000)
        sampled = df.iloc[::step][present].values.tobytes()
        content_hash = hashlib.sha256(sampled).hexdigest()[:12]
        # 纳入 adj_factor（复权因子变化必须使缓存失效）
        af_part = ""
        if "adj_factor" in df.columns:
            af_bytes = df[["adj_factor"]].iloc[::step].values.tobytes()
            af_part = hashlib.sha256(af_bytes).hexdigest()[:8]
        raw = f"{len(df)}|{df.index.min() if hasattr(df.index, 'min') else 0}|{df.index.max() if hasattr(df.index, 'max') else 0}|{content_hash}|{af_part}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _cache_root() -> Path:
    """计算缓存根目录，不依赖 prepare 模块避免循环导入。"""
    try:
        from UtilsManager.ConfigParser import Config
        base = Path(Config().CACHE_DIRECTORY) / "backtest_signal_cache"
    except Exception:
        base = Path(__file__).resolve().parent / "data" / "signal_cache"
    return base / "indicator_cache_v1"


def _indicators_path(symbol: str) -> Path:
    cr = _cache_root()
    bucket = symbol[:2].lower()
    (cr / bucket).mkdir(parents=True, exist_ok=True)
    return cr / bucket / f"{symbol}.indicators.parquet"


def _peaks_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.peaks.npy"


def _troughs_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.troughs.npy"


def _meta_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.meta.json"


# ── 背离检测缓存（参数无关，Phase 0 预计算） ──────────────────────

def _divergence_path(symbol: str) -> Path:
    cr = _cache_root()
    return cr / symbol[:2].lower() / f"{symbol}.divergence.npz"


def precompute_divergence(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """因果逐 bar 背离检测（与 vectorized_signal._divergence_scores 完全一致）。

    结果仅依赖 DIF 数据，与任何评分参数无关，可安全跨参数迭代复用。
    """
    from BackTrading.vectorized_signal import _divergence_scores
    from LogicAnalyzer.signals.divergence import adaptive_distance

    _dd = adaptive_distance(df["DIF"], base_distance=10) if "DIF" in df.columns else 11
    return _divergence_scores(df, base_distance=_dd)


def _load_divergence_from_disk(symbol: str, expected_fp: str | None = None) -> bool:
    p = _divergence_path(symbol)
    if not p.exists():
        return False
    if expected_fp:
        meta_path = _divergence_path(symbol).with_suffix(".meta.json")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("fingerprint") != expected_fp:
            return False
    try:
        with np.load(p, allow_pickle=True) as z:
            _DIVERGENCE[symbol] = (
                z["div_type"], z["div_idx"], z["div_strength"],
            )
        return True
    except Exception:
        return False


def _save_divergence_to_disk(symbol: str, div: tuple[np.ndarray, np.ndarray, np.ndarray], fp: str | None = None) -> None:
    p = _divergence_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    if _os.write_mode() == _os.OUTPUT_WRITE_REPLACE:
        np.savez(p, div_type=div[0], div_idx=div[1], div_strength=div[2])
    else:
        _os.atomic_write_npz(p, div_type=div[0], div_idx=div[1], div_strength=div[2])
    if fp:
        try:
            _meta = _divergence_path(symbol).with_suffix(".meta.json")
            if _os.write_mode() == _os.OUTPUT_WRITE_REPLACE:
                _meta.write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
            else:
                _os.atomic_write_text(_meta, json.dumps({"fingerprint": fp}, ensure_ascii=False))
        except Exception:
            pass


def get_divergence(
    symbol: str,
    df: pd.DataFrame | None = None,
    stock_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """获取预计算的背离检测结果（内存 → 磁盘 → 实时计算兜底）。

    与 get_precomputed 的 fallback 语义一致；df 为空时仅查缓存。
    计算失败时返回 None，由调用方回退到逐 bar 实时计算。
    """
    if symbol in _DIVERGENCE and (
        _SYMBOL_FPS.get(symbol) == _ACTIVE_FINGERPRINT or _ACTIVE_FINGERPRINT is None
    ):
        return _DIVERGENCE[symbol]
    _fp = _data_fingerprint(df) if df is not None and not df.empty else None
    if _load_divergence_from_disk(symbol, expected_fp=_fp):
        if _fp is not None:
            _SYMBOL_FPS[symbol] = _fp
        return _DIVERGENCE[symbol]
    if df is None or df.empty:
        return None
    try:
        div = precompute_divergence(df)
    except Exception:
        return None
    _DIVERGENCE[symbol] = div
    if _fp is not None:
        _SYMBOL_FPS[symbol] = _fp
    try:
        _save_divergence_to_disk(symbol, div, fp=_fp)
    except Exception:
        pass
    return div


def _pipeline_version() -> str:
    """管线代码版本（lazy import 避免循环导入）。

    指标计算逻辑变更（prepare/vectorized_signal/indicator_cache 等文件 mtime
    变化）时自动失效磁盘指标缓存，防止旧口径指标被复用（P0-2 空间统一）。
    """
    try:
        from BackTrading.prepare import _pipeline_version_hash
        return _pipeline_version_hash()
    except Exception:
        return "unknown"


def _load_from_disk(symbol: str, expected_fp: str | None = None) -> bool:
    """从磁盘加载到内存缓存。

    expected_fp: 当前输入数据的指纹，与 meta 中保存的指纹不一致时
    判定缓存失效（数据日期范围/内容已变化），强制重算，防止窗口切片
    缓存污染全量调用（WFO 场景 2025-11 后信号全 0 的根因）。
    管线版本（meta["version"]）与当前代码不一致时同样判定失效。
    """
    ipath = _indicators_path(symbol)
    ppath = _peaks_path(symbol)
    tpath = _troughs_path(symbol)
    if not (ipath.exists() and ppath.exists() and tpath.exists()):
        return False
    try:
        meta = json.loads(_meta_path(symbol).read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    if expected_fp and meta.get("fingerprint") != expected_fp:
        return False
    if meta.get("version") != _pipeline_version():
        return False
    # 参数无关特征列依赖的配置常量（slope_window）变化时同样失效，
    # 否则会静默复用旧窗口的斜率分（指标缓存 key 不含 config_hash）。
    if meta.get("feat_const") != _p0_constants():
        return False
    try:
        _IN_MEMORY[symbol] = pd.read_parquet(ipath)
        _IN_MEMORY[symbol].attrs["_p0_feat_const"] = meta.get("feat_const") or {}
        _PEAKS[symbol] = np.load(ppath)
        _TROUGHS[symbol] = np.load(tpath)
        return True
    except Exception:
        return False


def _save_to_disk(symbol: str, df: pd.DataFrame, peaks: np.ndarray, troughs: np.ndarray,
                  precheck_reasons: list[str] | None = None) -> None:
    meta = {
        "fingerprint": _data_fingerprint(df),
        "n_rows": len(df),
        "version": _pipeline_version(),
        # 参数无关特征列的计算常量（slope_window），加载时校验一致性
        "feat_const": _p0_constants(),
    }
    if precheck_reasons:
        meta["precheck"] = precheck_reasons
    if _os.write_mode() == _os.OUTPUT_WRITE_REPLACE:
        # 回退：禁用 upsert，直接替换写（分片前原始行为）
        df.to_parquet(_indicators_path(symbol), index=False, compression="zstd", compression_level=3)
        np.save(_peaks_path(symbol), peaks)
        np.save(_troughs_path(symbol), troughs)
        with open(_meta_path(symbol), "w") as f:
            json.dump(meta, f)
        return
    # upsert：原子写（tmp + os.replace），同 key 覆写，无半成品文件
    _os.atomic_write_parquet(_indicators_path(symbol), df)
    _os.atomic_write_npy(_peaks_path(symbol), peaks)
    _os.atomic_write_npy(_troughs_path(symbol), troughs)
    _os.atomic_write_text(_meta_path(symbol), json.dumps(meta, ensure_ascii=False))


def _precompute_divergences_parallel(symbols: list[str]) -> None:
    """并行预计算背离检测（背离仅依赖 DIF，与参数无关，跨迭代复用）。"""
    from concurrent.futures import ThreadPoolExecutor

    workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(get_divergence, s, _IN_MEMORY[s]) for s in symbols]
        for fut in futures:
            try:
                fut.result()
            except Exception:
                pass


def _precompute_one_symbol(f: Path, symbol: str, suspension_stats: dict[str, Any] | None = None) -> str:
    """Phase 0 单只股票指标预计算（片内串行、片间并行的最小执行单元）。

    返回: computed=本次新计算 / cached=磁盘缓存命中 / skipped=内存已有
    / empty=数据不足或预检跳过（写入空缓存）。
    """
    if symbol in _IN_MEMORY:
        return "skipped"

    df_raw = pd.read_parquet(f)
    _fp = _data_fingerprint(df_raw)
    if _load_from_disk(symbol, expected_fp=_fp):
        _SYMBOL_FPS[symbol] = _fp
        return "cached"

    if len(df_raw) < 60:
        _IN_MEMORY[symbol] = pd.DataFrame()
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        _SYMBOL_FPS[symbol] = _fp
        return "empty"

    # ── 窗口预检（指标计算前）：SKIP → 跳过并写快照；NEED_FILL → 限界填充 ──
    # Task F: 日历口径停牌统计（停牌占比超阈值 → SKIP），无统计时回退启发式
    from BackTrading.precheck import apply_precheck as _apply_precheck
    from BackTrading.prepare import _compute_indicators_snapshotted

    _susp = (suspension_stats or {}).get(symbol)
    df_raw, _pre_res = _apply_precheck(symbol, df_raw, context="precompute_all_indicators",
                                       suspension_stats=_susp)
    if df_raw.empty:
        _IN_MEMORY[symbol] = pd.DataFrame()
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        _SYMBOL_FPS[symbol] = _fp
        return "empty"
    _pre_reasons = _pre_res.reasons if _pre_res.status.value != "OK" else None

    df_ind = _compute_indicators_snapshotted(
        df_raw, symbol=symbol, context="precompute_all_indicators"
    )
    # 参数无关特征（评分层跨试次复用）：一次性计算并随指标缓存落盘
    df_ind = _p0_features(df_ind)

    _IN_MEMORY[symbol] = df_ind
    _PEAKS[symbol] = np.array([], dtype=int)
    _TROUGHS[symbol] = np.array([], dtype=int)
    _SYMBOL_FPS[symbol] = _fp
    _save_to_disk(symbol, df_ind, np.array([], dtype=int), np.array([], dtype=int),
                  precheck_reasons=_pre_reasons)
    return "computed"


def precompute_all_indicators(stock_dir: str, fingerprint: str | None = None,
                              shard_mode: str | None = None,
                              max_workers: int = 0,
                              batch_size: int = 0,
                              checkpoint_dir: str | None = None,
                              suspension_stats: dict[str, Any] | None = None) -> None:
    """Phase 0: 为 stock_dir 中所有股票预计算技术指标（不含 peaks/troughs）。

    peaks/troughs 改为在 _divergence_scores 中滚动计算以避免未来函数。
    背离检测与参数无关，在此并行预计算并落盘（内存 → 磁盘 → 实时计算三级缓存）。
    写入磁盘缓存 + 内存缓存。幂等。

    fingerprint: 当前整批数据（股票池+日期区间）的指纹。与上一批不一致时清空
    内存缓存，防止 WFO 跨窗口复用旧切片指标（2026-08-07 OOS 0 交易根因）。

    D1 分片（shard_mode=None → 读 Config().SHARD_MODE）：
    - off: 单任务串行执行（分片前原始行为，零 checkpoint 写入）。
    - symbol/hybrid: 按股票分批，ThreadPoolExecutor 并发；checkpoint 断点续跑
      （同 fingerprint 的 DONE 片跳过），失败片在下次调用时仅重跑失败片。
    - 失败语义保持分片前契约：单次尝试后原样重抛首个失败片异常（快照一次），
      由 checkpoint 保证跨调用续跑只重跑失败片。
    suspension_stats: Task F 日历口径停牌统计 {symbol: stats}，超阈值股票
    在指标计算前 SKIP（precheck 日历口径硬拒）。
    """
    global _ACTIVE_FINGERPRINT
    if fingerprint is None or fingerprint != _ACTIVE_FINGERPRINT:
        _reset_memory_caches()
        _ACTIVE_FINGERPRINT = fingerprint

    stock_files = sorted(Path(stock_dir).glob("*.parquet"))
    if not stock_files:
        logger.warning("Phase 0: stock_dir 中无 parquet 文件，跳过预计算")
        return

    from BackTrading import sharding as _sh

    _set = _sh.shard_settings()
    mode = shard_mode if shard_mode is not None else _set["mode"]
    max_workers = max_workers or _set["max_workers"]
    batch_size = batch_size or _set["batch_size"]
    checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else _set["checkpoint_dir"]

    symbols = [f.stem for f in stock_files]
    specs = _sh.shard_specs(
        symbols, mode=mode, batch_size=batch_size, dimension=_sh.SHARD_DIM_SYMBOL
    )

    def _worker(spec: _sh.ShardSpec) -> list[str]:
        _dir = Path(stock_dir)
        return [_precompute_one_symbol(_dir / f"{sym}.parquet", sym, suspension_stats)
                for sym in spec.keys]

    task_id = "phase0_" + hashlib.sha1((fingerprint or "no_fp").encode()).hexdigest()[:12]
    report = _sh.run_shards(
        specs, _worker,
        task_id=task_id,
        fingerprint=fingerprint or "no_fp",
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        max_workers=max_workers,
        max_attempts=1,  # 单次尝试：失败原样重抛（快照一次）；续跑重试由 checkpoint 覆盖
        dimension=_sh.SHARD_DIM_SYMBOL,
    )

    computed = sum(1 for _sid, r in report.results for o in r if o == "computed")
    cached = sum(1 for _sid, r in report.results for o in r if o == "cached")
    skipped = sum(1 for _sid, r in report.results for o in r if o == "skipped")
    empty = sum(1 for _sid, r in report.results for o in r if o == "empty")

    # 断点续跑回填：同 fingerprint 被跳过的 DONE 片从磁盘缓存载回内存，
    # 保证本进程后续调用（含背离预计算）能看到完整股票池。
    if report.skipped_ids and mode != "off":
        _by_id = {s.shard_id: s for s in specs}
        for _sid in report.skipped_ids:
            _spec = _by_id.get(_sid)
            if _spec is None:
                continue
            for _sym in _spec.keys:
                if _sym not in _IN_MEMORY:
                    get_precomputed(_sym, stock_dir)

    # 背离检测结果只依赖 DIF、与评分参数无关：Phase 0 并行预计算并落盘，
    # 后续每轮贝叶斯迭代（不同参数）直接复用，避免重跑 O(n²) 逐 bar 循环。
    # 缺失背离的股票（含磁盘加载 / 长 K 线股票）一次补齐；已缓存的零开销跳过。
    missing = [
        s for s in _IN_MEMORY
        if s not in _DIVERGENCE and len(_IN_MEMORY[s]) >= 60
    ]
    if missing:
        _t0 = time.perf_counter()
        _precompute_divergences_parallel(missing)
        logger.info(
            f"Phase 0: 背离预计算 {len(missing)} 只耗时 {time.perf_counter() - _t0:.1f}s"
        )

    # ── Task E 幂等输出：片输出按 (shard_id, key) upsert 到表级清单 + 合并校验 ──
    if _os.write_mode() == _os.OUTPUT_WRITE_UPSERT:
        try:
            _schema = _os.OutputSchema("indicator_cache", ("symbol",), "v1")
            _manifest = _os.OutputManifest(_cache_root(), _schema, batch_id=task_id)
            _by_id = {s.shard_id: s for s in specs}
            _records: list[_os.OutputRecord] = []
            for _sid, _outcomes in report.results:
                _spec = _by_id.get(_sid)
                if _spec is None:
                    continue
                for _sym, _outcome in zip(_spec.keys, _outcomes):
                    if _outcome == "skipped":
                        continue
                    _records.append(_os.OutputRecord(
                        shard_id=_sid,
                        key=_sym,
                        path=str(_indicators_path(_sym)) if _outcome in ("computed", "cached") else "",
                        rows=len(_IN_MEMORY.get(_sym, pd.DataFrame())),
                        fingerprint=_SYMBOL_FPS.get(_sym, ""),
                        written_at=time.time(),
                    ))
            _manifest.upsert_many(_records)
            _merge = _os.validate_artifacts(_manifest.records())
            _os.log_merge(_merge, "Phase 0 指标")
        except Exception as _e:
            logger.warning(f"[output] Phase 0 清单写入失败（不影响计算）: {_e}")

    _log_msg = f"Phase 0: {len(stock_files)} 只股票"
    parts = []
    if computed:
        parts.append(f"+{computed}")
    if cached:
        parts.append(f"cache{cached}")
    if skipped:
        parts.append(f"mem{skipped}")
    if empty:
        parts.append(f"empty{empty}")
    if parts:
        _log_msg += " (" + "/".join(parts) + ")"
    if report.failed:
        _log_msg += f" 分片失败 {report.failed}: {','.join(report.failed_ids)}"
    if mode != "off" and report.skipped:
        _log_msg += f" 断点跳过 {report.skipped} 片"
    logger.info(_log_msg)


def get_precomputed(
    symbol: str,
    stock_dir: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """获取预计算的指标和 peak/trough。

    尝试顺序: 内存缓存 → 磁盘缓存 → 实时计算(fallback)。

    Returns:
        (indicator_df, peaks, troughs)
        若股票不足 60 根 K 线，返回 (空 DataFrame, [], [])。
    """
    # 1. 内存缓存（校验所属批次指纹，防止跨窗口切片污染）
    if symbol in _IN_MEMORY:
        # 空结果（precheck SKIP/数据不足）无过期风险，始终返回，
        # 无需校验指纹——避免指纹不匹配时 fallback 重新计算已跳过的股票
        if _IN_MEMORY[symbol].empty:
            return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]
        if _SYMBOL_FPS.get(symbol) == _ACTIVE_FINGERPRINT or _ACTIVE_FINGERPRINT is None:
            return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    # 2. 磁盘缓存（校验指纹：数据范围/内容变化时缓存失效，防止 WFO 窗口切片污染）
    if stock_dir is not None:
        fpath = os.path.join(stock_dir, f"{symbol}.parquet")
        if os.path.exists(fpath):
            _fp = _data_fingerprint(pd.read_parquet(fpath))
            if _load_from_disk(symbol, expected_fp=_fp):
                _SYMBOL_FPS[symbol] = _fp
                return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]
    elif _load_from_disk(symbol):
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    # 3. Fallback：实时计算（只在非向量化模式或无 Phase 0 时触发）
    if stock_dir is None:
        return pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)

    fpath = os.path.join(stock_dir, f"{symbol}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame(), np.array([], dtype=int), np.array([], dtype=int)

    df_raw = pd.read_parquet(fpath)
    if len(df_raw) < 60:
        _IN_MEMORY[symbol] = pd.DataFrame()
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        _SYMBOL_FPS[symbol] = _data_fingerprint(df_raw)
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]

    from BackTrading.prepare import _compute_indicators_snapshotted
    from BackTrading.precheck import apply_precheck as _apply_precheck

    df_raw, _pre_res = _apply_precheck(symbol, df_raw, context="get_precomputed")
    if df_raw.empty:
        _IN_MEMORY[symbol] = pd.DataFrame()
        _PEAKS[symbol] = np.array([], dtype=int)
        _TROUGHS[symbol] = np.array([], dtype=int)
        _SYMBOL_FPS[symbol] = _data_fingerprint(df_raw)
        return _IN_MEMORY[symbol], _PEAKS[symbol], _TROUGHS[symbol]
    _pre_reasons = _pre_res.reasons if _pre_res.status.value != "OK" else None

    df_ind = _compute_indicators_snapshotted(df_raw, symbol=symbol, context="get_precomputed")
    df_ind = _p0_features(df_ind)

    _IN_MEMORY[symbol] = df_ind
    _PEAKS[symbol] = np.array([], dtype=int)
    _TROUGHS[symbol] = np.array([], dtype=int)
    _SYMBOL_FPS[symbol] = _data_fingerprint(df_raw)
    _save_to_disk(symbol, df_ind, np.array([], dtype=int), np.array([], dtype=int),
                  precheck_reasons=_pre_reasons)
    return df_ind, np.array([], dtype=int), np.array([], dtype=int)
