from pathlib import Path


def test_nightly_entrypoints_exist():
    root = Path(__file__).parents[1]
    for name in ("composite_distill_train.py", "calibrate_composite_kd.py", "evaluate_nightly_model.py", "run_nightly_kd_sweep.py"):
        assert (root / "scripts" / name).is_file()
