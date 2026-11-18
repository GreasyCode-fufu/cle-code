"""R6-B 复检: 注册 4 个对比网络后委托给本仓库未改动的 train_cle_fair.py(与 Codex 的 run_train.py 同模式)。
用法: python run_one.py <原 train_cle_fair 的全部参数>
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import adapters_r6b  # noqa: E402
import cle_fair_common  # noqa: E402

cle_fair_common.MODEL_REGISTRY.update(adapters_r6b.REGISTRY)

if __name__ == "__main__":
    try:
        runpy.run_module("train_cle_fair", run_name="__main__")
    except SystemExit as e:  # train_cle_fair 结束时的正常退出
        if e.code not in (0, None):
            raise
