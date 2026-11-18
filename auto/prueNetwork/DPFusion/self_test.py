#!/usr/bin/env python3
"""One-shot self test for the standalone DPFusion package.

Checks, in order:
  1. every designed model can be constructed;
  2. the recommended SN checkpoint reproduces its published test Macro-F1;
  3. the LCN+SN decision-level fusion reproduces its published test Macro-F1.

Run:  ./run.sh python self_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

TEST_DIR = os.path.join(ROOT, "datasets", "cle_ac_v1", "test")
SN_CKPT = os.path.join(ROOT, "weights", "sn_best", "model_best.pth")
LCN_CKPT = os.path.join(ROOT, "weights", "lcn_best", "model_best.pth")

EXPECT = {"sn": 0.9740, "fusion": 0.9721}
TOL = 0.0006


def step1_build():
    import cle_fair_common as c
    print("[1/3] build models")
    for name in ("cle_convnext_sn", "cle_convnext_lcn_only", "cle_convnext_dpf"):
        m = c.build_model(name, 3)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        y = m(torch.randn(1, 3, 224, 224))
        assert tuple(y.shape) == (1, 3), name
        print(f"      {name:26s} {n:6.2f} M  forward OK")


def _run_eval(model: str, weights: str) -> float:
    out = subprocess.run([sys.executable, "evaluate.py", "--model", model,
                          "--weights", weights, "--data-path", TEST_DIR,
                          "--out-dir", os.path.join(ROOT, "eval_out", "selftest_" + model)],
                         cwd=ROOT, capture_output=True, text=True)
    import json
    rj = os.path.join(ROOT, "eval_out", "selftest_" + model, "results.json")
    if not os.path.exists(rj):
        print(out.stdout[-800:], out.stderr[-800:])
        raise RuntimeError(f"evaluate.py failed for {model}")
    return json.load(open(rj))["macro_f1"]


def step2_sn():
    print("[2/3] evaluate SN checkpoint")
    f1 = _run_eval("cle_convnext_sn", SN_CKPT)
    print(f"      Macro-F1 = {f1:.4f}  (expected {EXPECT['sn']:.4f})")
    assert abs(f1 - EXPECT["sn"]) < TOL, "SN result mismatch"


def step3_fusion():
    print("[3/3] decision-level fusion (LCN + SN)")
    out = subprocess.run([sys.executable, "scripts/predict_fusion.py",
                          "--lcn-weights", LCN_CKPT, "--sn-weights", SN_CKPT,
                          "--data-path", TEST_DIR,
                          "--out-dir", os.path.join(ROOT, "eval_out", "selftest_fusion")],
                         cwd=ROOT, capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.strip().startswith("macro_f1")]
    if not line:
        print(out.stdout[-800:], out.stderr[-800:])
        raise RuntimeError("fusion failed")
    f1 = float(line[0].split()[-1])
    print(f"      Macro-F1 = {f1:.4f}  (expected {EXPECT['fusion']:.4f})")
    assert abs(f1 - EXPECT["fusion"]) < TOL, "fusion result mismatch"


if __name__ == "__main__":
    if not os.path.isdir(TEST_DIR):
        sys.exit(f"[ERROR] test data not found: {TEST_DIR}")
    step1_build()
    step2_sn()
    step3_fusion()
    print("\nALL CHECKS PASSED")
