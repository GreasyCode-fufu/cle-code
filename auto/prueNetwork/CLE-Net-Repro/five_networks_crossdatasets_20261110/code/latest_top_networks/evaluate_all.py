"""Evaluate every completed recent-network seed on the held-out CLE_ac test set."""
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/login/data/cle/auto/prueNetwork/CLE-Net")
PYTHON = "/home/login/miniconda3/envs/cle_topnets/bin/python"
TEST = "/home/login/data/cle/auto/ZCode/Classification/datasets/cle_ac_v1/test"
MODELS = (
    "thinkingvit_deit", "cpubone_b2", "mambavision_t",
    "mambaout_tiny", "transnext_tiny",
)

for model in MODELS:
    for seed in (42, 43, 44):
        run_dir = ROOT / "latest_top_networks" / "results" / model / f"seed{seed}" / model
        weights = run_dir / "model_best.pth"
        if not weights.exists():
            print(f"SKIP missing {weights}")
            continue
        out_dir = run_dir / "eval_test"
        cmd = [
            PYTHON, str(ROOT / "latest_top_networks" / "run_eval.py"),
            "--model", model,
            "--weights", str(weights),
            "--data-path", TEST,
            "--run-meta", str(run_dir / "run_meta.json"),
            "--out-dir", str(out_dir),
            "--batch-size", "64",
        ]
        print("RUN", model, seed, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
