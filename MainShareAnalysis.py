from __future__ import annotations

import os
import sys

# 禁用 pandera 导入警告
os.environ['DISABLE_PANDERA_IMPORT_WARNING'] = 'True'

import warnings
warnings.filterwarnings("ignore", message="lbfgs failed to converge")
warnings.filterwarnings("ignore", message="is close to the specified")
warnings.filterwarnings("ignore", message="The optimal value found")
import pandas as pd
# 全局禁用 PyArrow 后端，防止 0xC0000005 访问违例
pd.options.future.infer_string = False
pd.options.mode.string_storage = "python"

from loguru import logger

from Review.coordinator import StockAnalysisCoordinatorFactory


def main() -> None:
    """BAISYS_QUANT 统一入口 — 回测校准 + 每日复盘一体化。

    CLI 参数:
        --force              强制重新回测（忽略频率检查）
        --pipeline-only      仅执行每日复盘管线，跳过回测
        --backtest-only      仅执行回测，跳过每日复盘
        --schedule           启动回测定时调度器（常驻进程）
    """
    import io

    # 强制 Windows UTF-8 输出，防止 UnicodeEncodeError
    if sys.platform.startswith("win"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    # ── 初始化文件日志（必须先于所有业务代码） ──

    # 改为 {PROJECT_ROOT}/logs/，增加 LOG_DIR 环境变量覆盖，确保目录不存在时自动创建。
    import os as _os
    from pathlib import Path as _Path
    PROJECT_ROOT = str(_Path(__file__).resolve().parent)
    _log_dir = _os.environ.get("LOG_DIR", _os.path.join(PROJECT_ROOT, "logs"))
    _os.makedirs(_log_dir, exist_ok=True)
    logger.add(
        _os.path.join(_log_dir, "Corenews_Main.log"),
        level="INFO", encoding="utf-8", enqueue=True, rotation="1 day",
    )

    # 原实现全局 monkeypatch 第三方 AShareHub.__init__ 并 verify=False 禁用 TLS
    # 企业代理自签名证书场景：导出代理 CA 为 PEM 并设置 SSL_CERT_FILE 即可。
    logger.info("AShareHub TLS 校验已启用（仅信任显式 CA 或系统信任库）")

    # ── 额外回测专用日志文件（时间后缀，方便定位回测问题） ──
    from datetime import datetime as _dt
    _bt_log_name = f"backtest_{_dt.now().strftime('%Y%m%d_%H%M%S')}.log"
    _bt_log_path = _os.path.join(_log_dir, _bt_log_name)
    logger.add(_bt_log_path, level="DEBUG", encoding="utf-8", enqueue=True,
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}")
    logger.info(f"回测日志: {_bt_log_path}")

    args = [a.lstrip("-").replace("-", "_") for a in sys.argv[1:]]
    force = "force" in args
    pipeline_only = "pipeline_only" in args
    backtest_only = "backtest_only" in args
    schedule = "schedule" in args

    logger.info("=" * 80)
    logger.info("BAISYS_QUANT 量化复盘分析系统")
    logger.info("=" * 80)

    # ── 回测定时调度器 ──────────────────────────────────────
    if schedule:
        from BackTrading.runner import start_scheduler

        start_scheduler()
        return

    # ── 回测校准阶段 ────────────────────────────────────────
    if not pipeline_only:
        from BackTrading.calibration_log import ensure_table, get_last_run, should_rerun
        from UtilsManager.ConfigParser import Config
        from DataManager.DbEngine import get_engine

        cfg = Config()
        bt = cfg.app_config.backtest

        if bt.ENABLED:
            engine = get_engine(cfg)
            ensure_table(engine)

            last = get_last_run(engine)
            should, reason = should_rerun(last, bt.OPTIMIZE_FREQUENCY)

            if should or force:
                if force:
                    logger.info("--force 指定，强制回测校准")
                else:
                    logger.info(reason)
                    logger.info("到期，校准完成后自动进入复盘流程")

                from BackTrading.runner import run_backtest_pipeline

                result = run_backtest_pipeline(cfg, force=True)
                if result is None:
                    logger.warning("回测未完成，使用现有参数继续复盘")
                else:
                    logger.info("回测校准完成，参数已写入 config.ini")
                    # 重新加载配置，让复盘模块使用最新参数
                    cfg = Config()
            else:
                logger.info(f"回测未到期（{reason}），跳过校准，直接复盘")
        else:
            logger.info("回测未启用 (BACKTEST.enabled=false)，跳过校准")

    # ── 每日复盘阶段 ────────────────────────────────────────
    if not backtest_only:
        logger.info("")
        try:
            coordinator = StockAnalysisCoordinatorFactory.create(
                config_file="config.ini",
                force_rerun=force,
            )
            coordinator.run()

            logger.info("")
            logger.info("=" * 80)
            logger.info("=" * 80)
            logger.info("   - Excel报告: temp_data/审计报告_YYYYMMDD.xlsx")
            logger.info("   - 日志文件: logs/Corenews_Main_YYYYMMDD.log")

        except Exception as e:
            logger.error("")
            logger.error("=" * 80)
            logger.error(f"[FAIL] 分析流程失败: {type(e).__name__}")
            logger.error(f"   错误信息: {e}")
            logger.error("=" * 80)
            raise


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    import multiprocessing as _mp
    if _mp.current_process().name != "MainProcess":
        pass  # WFO / grid search 子进程 worker，跳过主入口
    else:
        main()
