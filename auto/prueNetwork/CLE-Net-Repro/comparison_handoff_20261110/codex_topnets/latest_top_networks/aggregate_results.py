"""Aggregate test metrics and validation selections across three seeds."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/login/data/cle/auto/prueNetwork/CLE-Net/latest_top_networks")
MODELS = (
    "thinkingvit_deit", "cpubone_b2", "mambavision_t",
    "mambaout_tiny", "transnext_tiny",
)

rows = []
for model in MODELS:
    for seed in (42, 43, 44):
        run = ROOT / "results" / model / f"seed{seed}" / model
        result_path = run / "eval_test" / "results.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
        metrics = result.get("metrics", result)
        complexity = result.get("complexity", {})
        with (run / "train_log.csv").open(newline="", encoding="utf-8") as f:
            val_best = max(float(x["val_macro_f1"]) for x in csv.DictReader(f))
        params = complexity.get("params")
        macs = complexity.get("macs")
        if model == "thinkingvit_deit":
            # THOP only counts modules exercised by one nested subnetwork and therefore
            # under-reports this dynamic model. Use static parameters and the official
            # threshold=0.0 full-second-stage GMAC figure from the authors' README.
            params, macs = "21.74M", "5.85G"
        rows.append({
            "model": model,
            "seed": seed,
            "val_best_macro_f1": val_best,
            "test_macro_f1": metrics.get("macro_f1"),
            "test_accuracy": metrics.get("accuracy"),
            "test_macro_auroc": metrics.get("macro_auroc", metrics.get("auc_roc_macro")),
            "test_macro_sensitivity": metrics.get("macro_sensitivity"),
            "test_macro_specificity": metrics.get("macro_specificity"),
            "params": params,
            "macs": macs,
        })

csv_path = ROOT / "comparison_per_seed.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = []
for model in MODELS:
    selected = [r for r in rows if r["model"] == model]
    if not selected:
        continue
    item = {"model": model, "n_seeds": len(selected)}
    for key in ("test_macro_f1", "test_accuracy", "test_macro_auroc",
                "test_macro_sensitivity", "test_macro_specificity"):
        vals = np.asarray([r[key] for r in selected if r[key] is not None], dtype=float)
        item[key + "_mean"] = float(vals.mean()) if len(vals) else None
        item[key + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    item["params"] = selected[0]["params"]
    item["macs"] = selected[0]["macs"]
    summary.append(item)

(ROOT / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with (ROOT / "comparison_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(json.dumps(summary, indent=2, ensure_ascii=False))
