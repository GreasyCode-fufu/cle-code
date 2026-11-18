"""
CLE fair-comparison EVALUATION script (classifyML) — aligned to the CUFIT eval protocol.
New file; does NOT modify evaluate_2class.py.

Pipeline:
  1. Load the best checkpoint (--model rebuilds the architecture, --weights = model_best.pth).
  2. Inference on VAL -> sweep threshold (0.05..0.95) maximizing Youden's J (sens+spec-1)
     on the positive-class probability -> best_thr.
  3. Inference on TEST -> report the FULL metric set TWICE:
       (a) threshold-optimized (prob_pos >= best_thr)
       (b) default 0.5 / argmax
     Metrics: Accuracy, Precision(macro), Recall(macro), F1(macro), AUROC, Sensitivity,
     Specificity, AUROC(weighted), AUC-PR, Balanced Accuracy, Cohen's Kappa, MCC,
     W-Precision, W-Recall, W-F1, Paper F1, PPV, NPV, Per-Class metrics, Confusion Matrix.
  4. Normalization is read from run_meta.json next to the weights (the dataset
     mean/std used in training), guaranteeing train/eval alignment; can be overridden
     with --norm-mean/--norm-std.
Usage:
  python evaluate_cle_fair.py --model fast_t0 -w ./weights/fast_t0/model_best.pth \
      --data-path-val /path/cle/val --data-path-test /path/cle/test --img_size 224
"""
import os
import json
import argparse

import numpy as np
import torch

from cle_fair_common import (
    build_model, read_images_dir, build_val_transform,
    run_inference, youden_threshold, compute_binary_metrics,
    compute_per_class_metrics, MODEL_REGISTRY,
)


def _resolve_norm(args):
    """Priority: explicit CLI mean/std > run_meta.json next to --weights."""
    if args.norm_mean is not None and args.norm_std is not None:
        return list(args.norm_mean), list(args.norm_std), "cli"
    meta_path = args.run_meta or os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                                              "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta["norm_mean"], meta["norm_std"], meta_path
    raise ValueError(
        "normalization not found. Pass --norm-mean/--norm-std, or ensure run_meta.json "
        f"exists next to the weights (looked at: {meta_path}).")


# ──────────────── CUFIT-style boxed report ────────────────

HR  = "+--------------------------------------------------------------+"
BL  = "|                                                              |"
SEP = "|--------------------------------------------------------------|"


def _print_boxed_report(title, m, per_class, class_names):
    """Print the full CUFIT-style boxed evaluation report."""
    tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
    sens = m['sensitivity']
    spec = m['specificity']

    # Per-class precision / recall / f1 from confusion matrix
    neg_precision = tn / (tn + fn + 1e-8)
    neg_recall    = tn / (tn + fp + 1e-8)
    neg_f1        = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + 1e-8)
    pos_precision = tp / (tp + fp + 1e-8)
    pos_recall    = tp / (tp + fn + 1e-8)
    pos_f1        = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + 1e-8)
    neg_n = tn + fp
    pos_n = fn + tp

    p = print

    print()
    p(HR)
    p("| {:<60s} |".format(title))
    p(HR)

    # --- Section 1: Per-Class ---
    p(BL)
    p("| Per-Class Metrics                                            |")
    p(SEP)
    p("| Class                  Prec   Recall       F1   AUROC  AUCPR |")
    p(SEP)
    p("| Negative (n={:<5})   {:.4f}   {:.4f}   {:.4f}  {:.4f} {:.4f} |".format(
        neg_n, neg_precision, neg_recall, neg_f1,
        per_class['negative']['auroc'], per_class['negative']['aucpr']))
    p("| Positive (n={:<5})   {:.4f}   {:.4f}   {:.4f}  {:.4f} {:.4f} |".format(
        pos_n, pos_precision, pos_recall, pos_f1,
        per_class['positive']['auroc'], per_class['positive']['aucpr']))
    p(SEP)

    # --- Section 2: Confusion Matrix ---
    p(BL)
    p("| Confusion Matrix                                             |")
    p(SEP)
    p("|                       Pred Neg    Pred Pos                   |")
    p(SEP)
    p("| Actual Neg (n={:<5})     {:<6}       {:<6}                 |".format(neg_n, tn, fp))
    p("| Actual Pos (n={:<5})     {:<6}       {:<6}                 |".format(pos_n, fn, tp))
    p(SEP)

    # --- Section 3: Clinical ---
    p(BL)
    p("| Clinical Metrics                                             |")
    p(SEP)
    p("| Sensitivity  (TP/(TP+FN))                          {:.4f}    |".format(m['sensitivity']))
    p("| Specificity  (TN/(TN+FP))                          {:.4f}    |".format(m['specificity']))
    p("| PPV          (TP/(TP+FP))                          {:.4f}    |".format(m['ppv']))
    p("| NPV          (TN/(TN+FN))                          {:.4f}    |".format(m['npv']))
    p(SEP)

    # --- Section 4: Summary ---
    p(BL)
    p("| Summary Metrics                                              |")
    p(SEP)
    p("| Accuracy                                           {:.4f}    |".format(m['accuracy']))
    p("| Precision (macro)                                  {:.4f}    |".format(m['precision_macro']))
    p("| Recall (macro)                                     {:.4f}    |".format(m['recall_macro']))
    p("| F1 (macro)                                         {:.4f}    |".format(m['f1_macro']))
    p("| AUROC                                              {:.4f}    |".format(m['auroc']))
    p("| Sensitivity                                        {:.4f}    |".format(m['sensitivity']))
    p("| Specificity                                        {:.4f}    |".format(m['specificity']))
    p("| AUROC (weighted)                                   {:.4f}    |".format(m['auroc_weighted']))
    p("| AUC-PR (positive class)                            {:.4f}    |".format(m['auc_pr']))
    p("| Balanced Accuracy                                  {:.4f}    |".format(m['balanced_accuracy']))
    p("| Cohen's Kappa                                      {:.4f}    |".format(m['kappa']))
    p("| MCC                                                {:.4f}    |".format(m['mcc']))
    p("| W-Precision                                        {:.4f}    |".format(m['precision_w']))
    p("| W-Recall                                           {:.4f}    |".format(m['recall_w']))
    p("| W-F1 (sklearn)                                     {:.4f}    |".format(m['f1_weighted']))
    p("| Paper F1 (harm mean Sns+Spc)                       {:.4f}    |".format(m['paper_f1']))
    p("| Macro F1                                           {:.4f}    |".format(m['f1_macro']))
    p(HR)
    print()


def _show_thr_opt(m, best_thr):
    """Print the concise threshold-optimized summary (CUFIT style)."""
    tn, fp, fn, tp = m['tn'], m['fp'], m['fn'], m['tp']
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv  = tp / (tp + fp + 1e-8)
    paperf1 = 2 * sens * spec / (sens + spec + 1e-8)

    print("\n" + "=" * 54)
    print("   THRESHOLD-OPTIMIZED TEST RESULTS (thr={:.3f})".format(best_thr))
    print("=" * 54)
    print(f"  AUROC          : {m['auroc']:.4f}")
    print(f"  Sensitivity    : {sens:.4f}")
    print(f"  Specificity    : {spec:.4f}")
    print(f"  PPV            : {ppv:.4f}")
    print(f"  Balanced Acc   : {m['balanced_accuracy']:.4f}")
    print(f"  MCC            : {m['mcc']:.4f}")
    print(f"  F1 (macro)     : {m['f1_macro']:.4f}")
    print(f"  Paper F1       : {paperf1:.4f}")
    print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print("=" * 54)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    norm_mean, norm_std, norm_src = _resolve_norm(args)
    print(f"[norm] mean={[round(x,4) for x in norm_mean]} "
          f"std={[round(x,4) for x in norm_std]}  (source={norm_src})")

    # ── build + load model ──
    model = build_model(args.model, num_classes=2)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    valid = set(model.state_dict().keys())
    state = {k: v for k, v in state.items() if k in valid}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"[model] {args.model} | weights={args.weights}")

    transform = build_val_transform(args.img_size, norm_mean, norm_std)

    # class order from val dir (sorted) -> negative=0, positive=1
    val_paths, val_labels, class_names = read_images_dir(args.data_path_val, args.class_names)
    test_paths, test_labels, _ = read_images_dir(args.data_path_test, class_names)
    assert len(class_names) == 2, f"fair eval is binary-only, got classes={class_names}"
    print(f"[data] classes={class_names} | val={len(val_paths)} test={len(test_paths)}")

    # ── inference ──
    val_prob = run_inference(model, val_paths, transform, args.batch_size, device)
    test_prob = run_inference(model, test_paths, transform, args.batch_size, device)

    # ── Youden's J threshold on VAL (positive-class prob) ──
    best_thr, best_j = youden_threshold(val_labels, val_prob[:, 1])
    print(f"\n[Threshold Opt] best_thr={best_thr:.3f} (Youden's J={best_j:.4f}) selected on VAL")

    # ── TEST predictions: two ways ──
    y_pred_opt = (test_prob[:, 1] >= best_thr).astype(int)
    y_pred_05 = (test_prob[:, 1] >= 0.5).astype(int)   # == argmax for binary

    # Per-class AUROC / AUCPR (threshold-free, same for both thresholds)
    per_class = compute_per_class_metrics(test_labels, test_prob)

    m_opt = compute_binary_metrics(test_labels, y_pred_opt, test_prob)
    m_05 = compute_binary_metrics(test_labels, y_pred_05, test_prob)

    # ── Print results ──
    _show_thr_opt(m_opt, best_thr)
    _print_boxed_report("TEST — threshold-optimized (Youden's J on val)",
                        m_opt, per_class, class_names)
    _print_boxed_report("TEST — default 0.5 / argmax",
                        m_05, per_class, class_names)

    # ── save ──
    out = {
        "model": args.model,
        "weights": os.path.abspath(args.weights),
        "img_size": args.img_size,
        "norm_mean": norm_mean, "norm_std": norm_std, "norm_source": str(norm_src),
        "class_names": class_names,
        "best_threshold": best_thr, "youden_j": best_j,
        "n_val": len(val_paths), "n_test": len(test_paths),
        "test_threshold_optimized": m_opt,
        "test_default_0.5": m_05,
        "per_class_metrics": per_class,
    }
    save_dir = args.out_dir or os.path.dirname(os.path.abspath(args.weights))
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "eval_cle_fair_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLE fair-comparison evaluation (binary)")
    parser.add_argument('--model', type=str, required=True,
                        help='must match training --model; one of: '
                             + ", ".join(sorted(MODEL_REGISTRY.keys())))
    parser.add_argument('--weights', '-w', type=str, required=True,
                        help='best checkpoint path (e.g. ./weights/<model>/model_best.pth)')
    parser.add_argument('--data-path-val', type=str, required=True,
                        help='patient-level val split dir (for threshold search)')
    parser.add_argument('--data-path-test', type=str, required=True,
                        help='patient-level test split dir (for final metrics)')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--class-names', type=str, nargs='+', default=None,
                        help='default: sorted subdir names (negative, positive)')
    parser.add_argument('--norm-mean', type=float, nargs=3, default=None,
                        help='override; default reads run_meta.json next to weights')
    parser.add_argument('--norm-std', type=float, nargs=3, default=None)
    parser.add_argument('--run-meta', type=str, default=None,
                        help='path to run_meta.json (default: next to weights)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='where to save eval_cle_fair_results.json (default: weights dir)')
    parser.add_argument('--device', type=str, default='cuda:0')
    opt = parser.parse_args()
    main(opt)
