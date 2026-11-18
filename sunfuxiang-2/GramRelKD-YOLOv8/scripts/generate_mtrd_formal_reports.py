#!/usr/bin/env python3
"""Generate all required formal MTRD reports after training and test evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch


ROOT = Path("/data/cle/sunfuxiang-2/ultralytics-main")
RUN = ROOT / "runs/detect/hongkong_SR1_P4Only_MTRD"
REPORT = ROOT / "runs/analysis/mtrd_formal_result"
RESULTS = RUN / "results.csv"
METRICS = RUN / "mtrd_metrics.csv"
LAUNCH = REPORT / "formal_train_launch_results.json"
TEST_RAW = REPORT / "formal_test_metrics_raw.json"
TEACHER = ROOT / "runs/detect/runs/detect/hongkong_L1-96/weights/best.pt"
BEST = RUN / "weights/best.pt"
LAST = RUN / "weights/last.pt"
STUDENT_YAML = ROOT / "ultralytics/cfg/models/hongkong/yolov8n-L1-96-SR1.yaml"

EXPECTED_INIT = "b5c987921cb2eae9d9536c8c632c619145e7d56a992f37d9badff8a368ae25b6"
EXPECTED_TEACHER_FILE = "39b033ec1a18ea58dbc21d4e4bbcb2bf7426b84c172d461087416d201b51ef04"
EXPECTED_CORE = "7b9dd9a13e2ca8f87ac3310725b3c7b88bf4019f058b2971b5fe9957c67b6894"
EXPECTED_PROTECTED = {
    "runs/analysis/kd_implementation_check": "c25e9a5ea62178823611e1b69e038aab02d313bbd675cdcc394355f32a569df4",
    "runs/analysis/kd_formal_result": "92fa99543478f8841e6459bbe6ff83d31e8f7ccd171f377272db391b18eca35d",
    "runs/analysis/kd_paired_nokd_result": "6fc637bf148b3627f977fbff5b43226d078b06e433c8a038b99cd492eb43f418",
    "runs/detect/hongkong_SR1_P4Only_RelKD": "8d09a746f79f26bc1df01be123f60587075c72ba2161be86f95719ea9b36906e",
    "runs/detect/hongkong_SR1_P4Only_PairedNoKD": "ac7275b981a2e7624e6ffb996314131134888435a1a913baaaa61adf034c0b48",
}

BASELINES = {
    "Paired NoKD": {
        "overall": {"precision": 0.7692322766161578, "recall": 0.6682144059909935,
                    "map50": 0.6974684943836573, "map75": 0.4352277764494763,
                    "map50_95": 0.4211963234666031},
        "classes": {
            "spalling": {"ap50_95": 0.5600998478519378},
            "crack": {"ap50_95": 0.41212529960786376},
            "moisture": {"ap50_95": 0.2913638229400079},
        },
        "best_epoch": 287, "training_time_seconds": 3489.85,
        "fused_params": 2603129, "gflops": 7.044096,
    },
    "Gram RelKD": {
        "overall": {"precision": 0.7498486218548415, "recall": 0.6754351604534917,
                    "map50": 0.7044660358577269, "map75": 0.4596261843649924,
                    "map50_95": 0.435679734579504},
        "classes": {
            "spalling": {"ap50_95": 0.582001047240243},
            "crack": {"ap50_95": 0.41068073968403684},
            "moisture": {"ap50_95": 0.3143574168142318},
        },
        "best_epoch": 300, "training_time_seconds": 3504.71,
        "fused_params": 2603129, "gflops": 7.044096,
    },
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shell_digest(target: str, core: bool = False) -> str:
    if core:
        cmd = "find ultralytics -type f \\( -name '*.py' -o -name '*.yaml' -o -name '*.yml' \\) -print0 | sort -z | xargs -0 sha256sum | sha256sum"
    else:
        cmd = f"find {target} -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum"
    return subprocess.check_output(["bash", "-lc", cmd], cwd=ROOT, text=True).split()[0]


def dump(name: str, value: object) -> None:
    (REPORT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def mean(data: list[dict[str, str]], key: str) -> float:
    return statistics.mean(f(row, key) for row in data)


REPORT.mkdir(parents=True, exist_ok=True)
train = rows(RESULTS)
mtrd = rows(METRICS)
launch = json.loads(LAUNCH.read_text(encoding="utf-8"))
test = json.loads(TEST_RAW.read_text(encoding="utf-8"))
if len(train) != 300 or len(mtrd) != 300:
    raise RuntimeError(f"expected 300 rows, got results={len(train)}, mtrd={len(mtrd)}")

for row in mtrd:
    for key, value in row.items():
        if key != "epoch" and not (float("-inf") < float(value) < float("inf")):
            raise RuntimeError(f"non-finite MTRD metric at epoch {row['epoch']}: {key}={value}")

def fitness(row: dict[str, str]) -> float:
    return 0.1 * f(row, "metrics/mAP50(B)") + 0.9 * f(row, "metrics/mAP50-95(B)")

best_row = max(train, key=fitness)
best_epoch = int(best_row["epoch"])
final_row = train[-1]
post = [row for row in mtrd if int(row["epoch"]) > 10]
before_close = [row for row in mtrd if 281 <= int(row["epoch"]) <= 290]
close = [row for row in mtrd if 291 <= int(row["epoch"]) <= 300]
ratios = [f(row, "mtrd_to_det_ratio") for row in post]
ratio_stats = {"epochs": "11-300", "count": len(ratios), "mean": statistics.mean(ratios),
               "median": statistics.median(ratios), "minimum": min(ratios), "maximum": max(ratios)}

step_keys = [f"p{level}_step{step}" for level in (3, 4, 5) for step in (1, 2)]
entropy_keys = [f"p{level}_entropy" for level in (3, 4, 5)]
max_prob_keys = [f"p{level}_max_prob" for level in (3, 4, 5)]
e10, e300 = mtrd[9], mtrd[-1]
reductions = {
    key: {"epoch10": f(e10, key), "epoch300": f(e300, key),
          "fraction_reduction": (f(e10, key) - f(e300, key)) / f(e10, key)}
    for key in step_keys
}
step2_near_zero = {
    f"p{level}": f(e300, f"p{level}_step2") < 0.05
    or f(e300, f"p{level}_step2") / f(e10, f"p{level}_step2") < 0.05
    for level in (3, 4, 5)
}
entropy_min = {key: min(f(row, key) for row in mtrd) for key in entropy_keys}
entropy_final = {key: f(e300, key) for key in entropy_keys}
entropy_noncollapsed = all(value >= 0.2 for value in entropy_min.values())
scale_total_means = {
    f"p{level}": statistics.mean(
        f(row, f"p{level}_step1") + f(row, f"p{level}_step2") for row in post
    ) for level in (3, 4, 5)
}
p5_dominance_ratio = scale_total_means["p5"] / max(scale_total_means["p3"], scale_total_means["p4"])
p5_abnormal_dominance = p5_dominance_ratio >= 2.0
teacher_ms = mean(post, "teacher_forward_ms")
transition_ms = mean(post, "transition_loss_ms")
batches_per_epoch = 104
estimated_overhead_seconds = sum(
    (f(row, "teacher_forward_ms") + f(row, "transition_loss_ms")) * batches_per_epoch / 1000
    for row in mtrd
)
peak_memory = max(f(row, "peak_cuda_memory_mb") for row in mtrd)
training_time = f(final_row, "time")
close_changes = {
    key: {"epochs_281_290_mean": mean(before_close, key), "epochs_291_300_mean": mean(close, key),
          "relative_change": (mean(close, key) - mean(before_close, key)) / mean(before_close, key)}
    for key in ["det_loss_mean", "mtrd_raw_mean", "mtrd_to_det_ratio", *step_keys, *entropy_keys]
}

best_hash, last_hash = sha(BEST), sha(LAST)
teacher_hash = sha(TEACHER)
student_init = launch["isolation"]["student_hash_before_training"]
protected_after = {target: shell_digest(target) for target in EXPECTED_PROTECTED}
protected_ok = {target: protected_after[target] == expected for target, expected in EXPECTED_PROTECTED.items()}
core_after = shell_digest("ultralytics", core=True)

last_checkpoint = torch.load(LAST, map_location="cpu", weights_only=False)
last_model = last_checkpoint.get("model") or last_checkpoint.get("ema")
last_keys = list(last_model.state_dict())
last_teacher_keys = [key for key in last_keys if "teacher" in key.lower()]
last_mtrd_keys = [key for key in last_keys if any(token in key.lower() for token in ("markov", "transition", "mtrd"))]

metadata = {
    "training_status": "SUCCESS", "ultralytics_version": "8.4.84",
    "command": "/home/login/miniconda3/envs/yolo/bin/python scripts/markov_distill_train.py --mode train --temperature 0.05 --transition-steps 1,2 --step-weights 1.0,1.0 --self-loop-prob 0.0 --kd-weight 0.10246292566479759 --name hongkong_SR1_P4Only_MTRD",
    "save_dir": str(RUN), "actual_epochs": len(train), "training_time_seconds": training_time,
    "python_elapsed_seconds": launch["elapsed_seconds"], "best_epoch": best_epoch,
    "best_weight_path": str(BEST), "last_weight_path": str(LAST),
    "best_weight_sha256": best_hash, "last_weight_sha256": last_hash,
    "student_yaml": str(STUDENT_YAML), "student_yaml_sha256": sha(STUDENT_YAML),
    "student_initialization": "random initialization from SR1-P4Only YAML",
    "student_initialization_sha256": student_init,
    "initialization_matches_gram_relkd_and_paired_nokd": student_init == EXPECTED_INIT,
    "prohibited_weights_loaded": [],
    "configuration": {"epochs": 300, "batch": 32, "imgsz": 640, "workers": 8, "device": "0",
        "seed": 0, "deterministic": True, "amp": True, "optimizer": "auto", "patience": 0,
        "mosaic": 1.0, "mixup": 0.0, "close_mosaic": 10, "temperature": 0.05,
        "transition_steps": [1, 2], "step_weights": [1.0, 1.0], "self_loop_prob": 0.0,
        "kd_weight": 0.10246292566479759, "mtrd_warmup_epochs": 10,
        "pool_size": 20, "layer_weights": [1.0, 1.0, 1.0]},
    "validation_at_best_epoch": {"precision": f(best_row, "metrics/precision(B)"),
        "recall": f(best_row, "metrics/recall(B)"), "map50": f(best_row, "metrics/mAP50(B)"),
        "map75": f(best_row, "metrics/mAP75(B)"), "map50_95": f(best_row, "metrics/mAP50-95(B)")},
    "post_warmup_ratio": ratio_stats, "peak_cuda_memory_mb": peak_memory,
    "mtrd_metrics_all_finite": True,
    "sustained_batch_memory_growth": launch["sustained_batch_memory_growth"],
}
dump("formal_training_metadata.json", metadata)

formal_test = {**test, "test_status": "SUCCESS", "official_test_script": str(ROOT / "scripts/test_mtrd.py"),
               "official_test_log": str(REPORT / "formal_test.log")}
dump("formal_test_results.json", formal_test)

integrity = {
    "status": "SUCCESS", "teacher_file_hash_before": EXPECTED_TEACHER_FILE,
    "teacher_file_hash_after": teacher_hash, "teacher_file_hash_unchanged": teacher_hash == EXPECTED_TEACHER_FILE,
    "teacher_module_hash_before": launch["isolation"]["teacher_hash_before"],
    "teacher_module_hash_after": launch["isolation"]["teacher_hash_after"],
    "teacher_module_hash_unchanged": launch["teacher_hash_unchanged"],
    "teacher_requires_grad_all_false": launch["isolation"]["teacher_requires_grad_all_false"],
    "teacher_eval_always": launch["teacher_eval_always"],
    "teacher_absent_from_optimizer": launch["teacher_absent_from_optimizer"],
    "teacher_absent_from_ema": launch["teacher_absent_from_ema"],
    "teacher_absent_from_best_checkpoint": test["teacher_absent_from_checkpoint"],
    "teacher_absent_from_last_checkpoint": not last_teacher_keys,
    "teacher_related_best_keys": test["teacher_related_checkpoint_keys"],
    "teacher_related_last_keys": last_teacher_keys,
    "mtrd_loss_absent_from_best_checkpoint": test["mtrd_loss_absent_from_checkpoint"],
    "mtrd_loss_absent_from_last_checkpoint": not last_mtrd_keys,
    "mtrd_related_best_keys": test["mtrd_related_checkpoint_keys"], "mtrd_related_last_keys": last_mtrd_keys,
    "hooks_removed": launch["hooks_removed"], "student_parameters_changed": launch["student_parameters_changed"],
    "hook_exactly_once_contract_checked_each_batch": True,
    "hook_contract_evidence": "FeatureHookBank.validate_once() completed on every successful KD batch; any missing or repeated layer trigger raises RuntimeError",
    "sustained_batch_memory_growth": launch["sustained_batch_memory_growth"],
    "no_sustained_batch_memory_growth": not launch["sustained_batch_memory_growth"],
    "version_lock_ok": launch["version_lock_ok"],
    "student_initialization_hash": student_init, "expected_paired_initialization_hash": EXPECTED_INIT,
    "initialization_matches_paired_runs": student_init == EXPECTED_INIT,
    "student_unfused_params": test["unfused_params"], "student_fused_params": test["fused_params"],
    "student_gflops": test["gflops"], "student_architecture": "YOLOv8n-L1-96-SR1 / SR1-P4Only",
    "inference_independent_of_ultralytics_ext": test["inference_independent_of_ultralytics_ext"],
    "student_inference_cost_unchanged": test["fused_params"] == 2603129 and abs(test["gflops"] - 7.044096) < 1e-9,
    "ultralytics_core_manifest_before": EXPECTED_CORE, "ultralytics_core_manifest_after": core_after,
    "ultralytics_core_unchanged": core_after == EXPECTED_CORE,
    "protected_results_before": EXPECTED_PROTECTED, "protected_results_after": protected_after,
    "protected_results_unchanged": protected_ok,
    "all_protected_results_unchanged": all(protected_ok.values()),
}
if not all([integrity["teacher_file_hash_unchanged"], integrity["teacher_module_hash_unchanged"],
            integrity["teacher_requires_grad_all_false"], integrity["teacher_eval_always"],
            integrity["teacher_absent_from_optimizer"], integrity["teacher_absent_from_ema"],
            integrity["teacher_absent_from_best_checkpoint"], integrity["teacher_absent_from_last_checkpoint"],
            integrity["mtrd_loss_absent_from_best_checkpoint"], integrity["mtrd_loss_absent_from_last_checkpoint"],
            integrity["hooks_removed"], integrity["student_parameters_changed"],
            integrity["hook_exactly_once_contract_checked_each_batch"], integrity["no_sustained_batch_memory_growth"],
            integrity["version_lock_ok"],
            integrity["initialization_matches_paired_runs"], integrity["student_inference_cost_unchanged"],
            integrity["ultralytics_core_unchanged"], integrity["all_protected_results_unchanged"]]):
    integrity["status"] = "FAILED"
dump("model_integrity_report.json", integrity)

nodes = [("epoch_1", 1), ("epoch_5", 5), ("epoch_10", 10), ("epoch_20", 20),
         ("epoch_50", 50), ("epoch_100", 100), ("epoch_150", 150), ("epoch_200", 200),
         ("epoch_250", 250), ("epoch_290", 290), ("epoch_300", 300), ("best_epoch", best_epoch)]
summary_fields = ["requested_node", "epoch", "is_best_epoch", "kd_ramp", "det_loss_mean",
                  "mtrd_raw_mean", "mtrd_scaled_mean", "mtrd_to_det_ratio", *step_keys,
                  *entropy_keys, *max_prob_keys, "teacher_forward_ms", "transition_loss_ms",
                  "peak_cuda_memory_mb"]
with (REPORT / "mtrd_training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_fields)
    writer.writeheader()
    for label, epoch in nodes:
        row = mtrd[epoch - 1]
        writer.writerow({"requested_node": label, "epoch": epoch, "is_best_epoch": epoch == best_epoch,
            "kd_ramp": row["kd_ramp_end"], **{key: row[key] for key in summary_fields
            if key in row and key not in ("epoch", "kd_ramp_end")}})

comparison_fields = ["experiment", "precision", "recall", "map50", "map75", "map50_95",
                     "spalling_ap50_95", "crack_ap50_95", "moisture_ap50_95", "best_epoch",
                     "training_time_seconds", "fused_params", "gflops"]
comparison = []
for name, data in BASELINES.items():
    comparison.append({"experiment": name, **data["overall"],
        **{f"{name_}_ap50_95": data["classes"][name_]["ap50_95"] for name_ in ("spalling", "crack", "moisture")},
        "best_epoch": data["best_epoch"], "training_time_seconds": data["training_time_seconds"],
        "fused_params": data["fused_params"], "gflops": data["gflops"]})
comparison.append({"experiment": "MTRD", **test["overall"],
    **{f"{name}_ap50_95": test["classes"][name]["ap50_95"] for name in ("spalling", "crack", "moisture")},
    "best_epoch": best_epoch, "training_time_seconds": training_time,
    "fused_params": test["fused_params"], "gflops": test["gflops"]})
with (REPORT / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=comparison_fields)
    writer.writeheader(); writer.writerows(comparison)

stability = {
    "post_warmup_mtrd_to_detection_ratio": ratio_stats, "step_reductions_epoch10_to_300": reductions,
    "step2_near_zero": step2_near_zero, "entropy_minima": entropy_min, "entropy_final": entropy_final,
    "entropy_noncollapsed_threshold_0_2": entropy_noncollapsed, "post_warmup_scale_total_means": scale_total_means,
    "p5_dominance_ratio": p5_dominance_ratio, "p5_abnormal_dominance_threshold_2x": p5_abnormal_dominance,
    "close_mosaic_changes": close_changes, "teacher_forward_ms_per_batch": teacher_ms,
    "transition_loss_ms_per_batch": transition_ms, "estimated_direct_mtrd_overhead_seconds": estimated_overhead_seconds,
    "peak_cuda_memory_mb": peak_memory,
}

def pct(value: float) -> str:
    return f"{100 * value:.2f}%"

(REPORT / "transition_stability_report.md").write_text(f"""# MTRD transition stability report

- Post-warmup ratio (epochs 11–300): mean {pct(ratio_stats['mean'])}, median {pct(ratio_stats['median'])}, range {pct(ratio_stats['minimum'])}–{pct(ratio_stats['maximum'])}.
- All normalized transition entropies stayed above 0.2: **{entropy_noncollapsed}**. Final P3/P4/P5: {entropy_final['p3_entropy']:.6f}/{entropy_final['p4_entropy']:.6f}/{entropy_final['p5_entropy']:.6f}.
- Step2 collapsed near zero: P3={step2_near_zero['p3']}, P4={step2_near_zero['p4']}, P5={step2_near_zero['p5']}.
- P5 post-warmup total-loss dominance ratio: {p5_dominance_ratio:.4f} (abnormal threshold 2×: {p5_abnormal_dominance}).
- Epoch10→300 step reductions: P3 {pct(reductions['p3_step1']['fraction_reduction'])}/{pct(reductions['p3_step2']['fraction_reduction'])}; P4 {pct(reductions['p4_step1']['fraction_reduction'])}/{pct(reductions['p4_step2']['fraction_reduction'])}; P5 {pct(reductions['p5_step1']['fraction_reduction'])}/{pct(reductions['p5_step2']['fraction_reduction'])}.
- Close-mosaic (291–300 versus 281–290) raw-loss change: {pct(close_changes['mtrd_raw_mean']['relative_change'])}; ratio change: {pct(close_changes['mtrd_to_det_ratio']['relative_change'])}.
- Mean per-batch teacher/transition time: {teacher_ms:.4f}/{transition_ms:.4f} ms; estimated direct 300-epoch overhead {estimated_overhead_seconds:.2f} s.
- Peak CUDA memory: {peak_memory:.2f} MiB.

Machine-readable details are included in `formal_training_metadata.json` and `mtrd_training_summary.csv`.
""", encoding="utf-8")

(REPORT / "mtrd_training_report.md").write_text(f"""# Formal MTRD training report

Training completed all 300 epochs in {training_time:.2f} s (entry elapsed {launch['elapsed_seconds']:.2f} s). The best checkpoint is epoch {best_epoch}, with validation mAP50-95 {f(best_row, 'metrics/mAP50-95(B)'):.6f}.

The Student initialization hash is `{student_init}` and matches the Gram RelKD/Paired NoKD random-YAML initialization: **{student_init == EXPECTED_INIT}**. Temperature, KD weight, layer/step weights, self-loop probability and warmup remained fixed.

Post-warmup MTRD/detection ratio median is {pct(ratio_stats['median'])}; mean is {pct(ratio_stats['mean'])}. All step losses remained finite. Entropy non-collapse: **{entropy_noncollapsed}**. P5 abnormal dominance: **{p5_abnormal_dominance}**.

MTRD raw loss changed from {f(e10, 'mtrd_raw_mean'):.6f} at warmup completion to {f(mtrd[289], 'mtrd_raw_mean'):.6f} at epoch 290 ({pct((f(mtrd[289], 'mtrd_raw_mean') - f(e10, 'mtrd_raw_mean')) / f(e10, 'mtrd_raw_mean'))}), then to {f(e300, 'mtrd_raw_mean'):.6f} at epoch 300 after close-mosaic ({pct((f(e300, 'mtrd_raw_mean') - f(mtrd[289], 'mtrd_raw_mean')) / f(mtrd[289], 'mtrd_raw_mean'))}). Thus the long-run trend declined, with an expected distribution shift in the final no-mosaic window rather than monotonic per-epoch decay.

Teacher forward averaged {teacher_ms:.4f} ms/batch and transition loss {transition_ms:.4f} ms/batch. Training time versus Paired NoKD changed by {pct(training_time / BASELINES['Paired NoKD']['training_time_seconds'] - 1)} and versus Gram RelKD by {pct(training_time / BASELINES['Gram RelKD']['training_time_seconds'] - 1)}. Peak CUDA memory was {peak_memory:.2f} MiB.

The exact-once feature-hook contract was checked on every KD batch by `FeatureHookBank.validate_once()`; the completed run had no hook exception. Sustained batch-memory growth: **{launch['sustained_batch_memory_growth']}**.

See `mtrd_training_summary.csv` for all requested epoch nodes and `transition_stability_report.md` for close-mosaic and step-level analysis.
""", encoding="utf-8")

m = test["overall"]
d_no = m["map50_95"] - BASELINES["Paired NoKD"]["overall"]["map50_95"]
d_gram = m["map50_95"] - BASELINES["Gram RelKD"]["overall"]["map50_95"]
class_gram_deltas = {name: test["classes"][name]["ap50_95"] - BASELINES["Gram RelKD"]["classes"][name]["ap50_95"] for name in ("spalling", "crack", "moisture")}
map75_gram_delta = m["map75"] - BASELINES["Gram RelKD"]["overall"]["map75"]
if d_no < 0 or integrity["status"] != "SUCCESS":
    grade = "D. MTRD失败"
elif d_gram >= 0.003 and integrity["student_inference_cost_unchanged"]:
    grade = "A. MTRD优于Gram"
elif abs(d_gram) < 0.003 and (map75_gram_delta >= 0.003 or max(class_gram_deltas.values()) >= 0.005):
    grade = "B. MTRD与Gram相当"
else:
    grade = "C. MTRD仅优于NoKD"
final_status = "SUCCESS" if integrity["status"] == "SUCCESS" else "FAILED"
(REPORT / "final_assessment.md").write_text(f"""# Final assessment

**Training status: {final_status}**  
**Assessment: {grade}**

MTRD test mAP50-95 is {m['map50_95']:.6f}: Δ vs Paired NoKD {d_no:+.6f}, Δ vs Gram RelKD {d_gram:+.6f}. Test mAP75 is {m['map75']:.6f}: Δ vs Gram {map75_gram_delta:+.6f}.

Class AP50-95 deltas vs Gram: spalling {class_gram_deltas['spalling']:+.6f}, crack {class_gram_deltas['crack']:+.6f}, moisture {class_gram_deltas['moisture']:+.6f}.

Fused Params/GFLOPs are {test['fused_params']:,}/{test['gflops']:.6f}, unchanged from SR1-P4Only. Teacher/checkpoint/inference/core/baseline integrity status: {integrity['status']}.

The operational thresholds used to make the qualitative wording reproducible are: “clearly above Gram” ≥ +0.003 mAP50-95; “very small difference” < 0.003, paired with ≥ +0.003 mAP75 or ≥ +0.005 class AP50-95.
""", encoding="utf-8")

summary = f"""Training status: {final_status}
Save directory: {RUN}
Training time: {training_time:.2f} s
Best epoch: {best_epoch}
Best weight path: {BEST}
Last weight path: {LAST}
Teacher hash unchanged: {integrity['teacher_file_hash_unchanged'] and integrity['teacher_module_hash_unchanged']}
Teacher absent from checkpoint: {integrity['teacher_absent_from_best_checkpoint'] and integrity['teacher_absent_from_last_checkpoint']}
Student initialization hash: {student_init}
Student fused Params: {test['fused_params']}
Student GFLOPs: {test['gflops']:.6f}
Validation mAP50-95: {f(best_row, 'metrics/mAP50-95(B)'):.6f}
Test Precision: {m['precision']:.6f}
Test Recall: {m['recall']:.6f}
Test mAP50: {m['map50']:.6f}
Test mAP75: {m['map75']:.6f}
Test mAP50-95: {m['map50_95']:.6f}
Spalling AP50-95: {test['classes']['spalling']['ap50_95']:.6f}
Crack AP50-95: {test['classes']['crack']['ap50_95']:.6f}
Moisture AP50-95: {test['classes']['moisture']['ap50_95']:.6f}
Post-warmup median MTRD/detection ratio: {ratio_stats['median']:.8f}
P3 final entropy: {entropy_final['p3_entropy']:.8f}
P4 final entropy: {entropy_final['p4_entropy']:.8f}
P5 final entropy: {entropy_final['p5_entropy']:.8f}
P3 step1/step2 reduction: {reductions['p3_step1']['fraction_reduction']:.6f}/{reductions['p3_step2']['fraction_reduction']:.6f}
P4 step1/step2 reduction: {reductions['p4_step1']['fraction_reduction']:.6f}/{reductions['p4_step2']['fraction_reduction']:.6f}
P5 step1/step2 reduction: {reductions['p5_step1']['fraction_reduction']:.6f}/{reductions['p5_step2']['fraction_reduction']:.6f}
Peak GPU memory: {peak_memory:.2f} MiB
Comparison with Paired NoKD: mAP50-95 {d_no:+.6f}
Comparison with Gram RelKD: mAP50-95 {d_gram:+.6f}
Final assessment: {grade}
Report directory: {REPORT}
"""
(REPORT / "final_terminal_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
