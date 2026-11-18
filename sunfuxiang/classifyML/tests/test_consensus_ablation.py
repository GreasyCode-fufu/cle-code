import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from consensus_manifest import filter_training_samples
from scripts.summarize_consensus_ablation import (
    aggregate_seed_metrics,
    compute_metrics,
)
from train_cle_fair import (
    SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS,
    build_run_meta,
    load_pretrained_weights_strict,
)


class TinyPretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 3)


class StrictPretrainedLoadingTests(unittest.TestCase):
    def write_checkpoint(self, root, mutate=None):
        model = TinyPretrainedModel()
        state = dict(model.state_dict())
        if mutate is not None:
            mutate(state)
        path = Path(root) / "pretrained.pth"
        torch.save({"model": state}, path)
        return path

    def test_only_classifier_may_be_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_checkpoint(temporary)
            audit = load_pretrained_weights_strict(
                TinyPretrainedModel(), path, "unit_test_model"
            )
            self.assertEqual(audit["missing_keys"], ["head.bias", "head.weight"])
            self.assertEqual(audit["unexpected_keys"], [])
            self.assertTrue(audit["all_trainable_backbone_parameters_loaded"])

    def test_missing_backbone_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_checkpoint(
                temporary, lambda state: state.pop("backbone.bias")
            )
            with self.assertRaisesRegex(RuntimeError, "backbone loading mismatch"):
                load_pretrained_weights_strict(
                    TinyPretrainedModel(), path, "unit_test_model"
                )

    def test_unexpected_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            def add_unexpected(state):
                state["unexpected.weight"] = torch.zeros(1)

            path = self.write_checkpoint(temporary, add_unexpected)
            with self.assertRaisesRegex(RuntimeError, "backbone loading mismatch"):
                load_pretrained_weights_strict(
                    TinyPretrainedModel(), path, "unit_test_model"
                )

    def test_exact_vanilla_swin_mask_buffers_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            def add_buffers(state):
                for key in SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS:
                    state[key] = torch.zeros(1)

            path = self.write_checkpoint(temporary, add_buffers)
            audit = load_pretrained_weights_strict(
                TinyPretrainedModel(),
                path,
                "swin_tiny_vanilla_patch4_window7_224",
            )
            self.assertEqual(
                set(audit["ignored_nonparameter_buffer_keys"]),
                SWIN_VANILLA_IGNORED_CHECKPOINT_BUFFERS,
            )


class ManifestFilterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "train"
        self.val = Path(self.temp.name) / "val"
        self.test = Path(self.temp.name) / "test"
        self.class_names = [
            "Benign_lesions",
            "Malignant_lesions",
            "body_of_stomach_normal",
        ]
        for split in (self.root, self.val, self.test):
            for class_name in self.class_names:
                (split / class_name).mkdir(parents=True, exist_ok=True)

        self.samples = [
            ("Benign_lesions/train_a_S_aaa111.jpg", 0, "S_aaa111"),
            ("Malignant_lesions/train_b_S_bbb222.jpg", 1, "S_bbb222"),
            ("body_of_stomach_normal/train_c_S_ccc333.jpg", 2, "S_ccc333"),
        ]
        for relative, _, _ in self.samples:
            (self.root / relative).write_bytes(b"not-an-image")
        self.paths = [str(self.root / relative) for relative, _, _ in self.samples]
        self.labels = [label for _, label, _ in self.samples]

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows):
        path = Path(self.temp.name) / "subset.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["sample_id", "class_name", "relative_path"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_valid_manifest_filters_and_counts(self):
        manifest = self.write_manifest(
            [
                {
                    "sample_id": "S_aaa111",
                    "class_name": "Benign_lesions",
                    "relative_path": "Benign_lesions/train_a_S_aaa111.jpg",
                },
                {
                    "sample_id": "S_ccc333",
                    "class_name": "body_of_stomach_normal",
                    "relative_path": "body_of_stomach_normal/train_c_S_ccc333.jpg",
                },
            ]
        )
        paths, labels, info = filter_training_samples(
            self.paths,
            self.labels,
            self.root,
            manifest,
            exclusion_roots=[self.val, self.test],
        )
        self.assertEqual(paths, [self.paths[0], self.paths[2]])
        self.assertEqual(labels, [0, 2])
        self.assertEqual(info["actual_train_count"], 2)
        self.assertEqual(info["benign_count"], 1)
        self.assertEqual(info["malignant_count"], 0)
        self.assertEqual(info["normal_count"], 1)

    def test_duplicate_sample_id_is_rejected(self):
        row = {
            "sample_id": "S_aaa111",
            "class_name": "Benign_lesions",
            "relative_path": "Benign_lesions/train_a_S_aaa111.jpg",
        }
        manifest = self.write_manifest([row, dict(row)])
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            filter_training_samples(self.paths, self.labels, self.root, manifest)

    def test_parent_class_mismatch_is_rejected(self):
        manifest = self.write_manifest(
            [
                {
                    "sample_id": "S_aaa111",
                    "class_name": "Malignant_lesions",
                    "relative_path": "Benign_lesions/train_a_S_aaa111.jpg",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "class_name"):
            filter_training_samples(self.paths, self.labels, self.root, manifest)

    def test_unknown_or_outside_path_is_rejected(self):
        manifest = self.write_manifest(
            [
                {
                    "sample_id": "S_unknown",
                    "class_name": "Benign_lesions",
                    "relative_path": "../outside.jpg",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "outside train root"):
            filter_training_samples(self.paths, self.labels, self.root, manifest)

    def test_validation_or_test_overlap_is_rejected(self):
        (self.val / "Benign_lesions/val_S_aaa111.jpg").write_bytes(b"duplicate-id")
        manifest = self.write_manifest(
            [
                {
                    "sample_id": "S_aaa111",
                    "class_name": "Benign_lesions",
                    "relative_path": "Benign_lesions/train_a_S_aaa111.jpg",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "overlaps validation/test"):
            filter_training_samples(
                self.paths,
                self.labels,
                self.root,
                manifest,
                exclusion_roots=[self.val, self.test],
            )


class SummaryMetricTests(unittest.TestCase):
    def test_metrics_and_sample_standard_deviation(self):
        class_names = ["A", "B", "C"]
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 0])
        y_prob = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.7, 0.1],
                [0.1, 0.8, 0.1],
                [0.6, 0.1, 0.3],
            ]
        )
        metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("macro_auroc_ovr", metrics)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (3, 3))
        self.assertEqual(set(metrics["per_class"]), set(class_names))

        seeds = [
            {"macro_f1": 0.8, "accuracy": 0.7},
            {"macro_f1": 0.9, "accuracy": 0.8},
            {"macro_f1": 1.0, "accuracy": 0.9},
        ]
        aggregate = aggregate_seed_metrics(seeds)
        self.assertAlmostEqual(aggregate["macro_f1"]["mean"], 0.9)
        self.assertAlmostEqual(aggregate["macro_f1"]["std"], 0.1)
        self.assertEqual(aggregate["macro_f1"]["ddof"], 1)


class RunMetadataTests(unittest.TestCase):
    def test_no_manifest_metadata_keeps_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                model="resnet34",
                num_classes=3,
                seed=42,
                label_smoothing=0.0,
                data_path_train=str(root / "train"),
                data_path_val=str(root / "val"),
            )
            metadata = build_run_meta(
                args,
                img_size=224,
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225],
                norm_source="imagenet",
                save_path=root / "model.pth",
                manifest_info=None,
            )
            self.assertEqual(
                set(metadata),
                {
                    "model", "num_classes", "img_size", "seed",
                    "norm_mean", "norm_std", "norm_source",
                    "selection_metric", "loss", "data_path_train",
                    "data_path_val", "weights", "created_at",
                },
            )

    def test_manifest_run_metadata_contains_reproducibility_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pretrained = root / "pretrained.pth"
            pretrained.write_bytes(b"weights")
            manifest = root / "R.csv"
            manifest.write_text(
                "sample_id,class_name,relative_path\n"
                "S_a,Benign_lesions,Benign_lesions/a_S_a.jpg\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model="resnet34",
                num_classes=3,
                seed=42,
                label_smoothing=0.0,
                data_path_train=str(root / "train"),
                data_path_val=str(root / "val"),
                batch_size=32,
                epochs=300,
                lr=0.0001,
                early_stop=30,
                weights=str(pretrained),
            )
            (root / "train").mkdir()
            (root / "val").mkdir()
            manifest_info = {
                "subset": "R",
                "manifest_path": str(manifest),
                "manifest_sha256": "manifest-hash",
                "actual_train_count": 1,
                "normal_count": 0,
                "benign_count": 1,
                "malignant_count": 0,
            }
            metadata = build_run_meta(
                args,
                img_size=224,
                norm_mean=[0.485, 0.456, 0.406],
                norm_std=[0.229, 0.224, 0.225],
                norm_source="imagenet",
                save_path=root / "model.pth",
                manifest_info=manifest_info,
            )
            required = {
                "subset", "manifest_path", "manifest_sha256",
                "actual_train_count", "normal_count", "benign_count", "malignant_count",
                "model", "seed", "batch_size", "max_epochs", "optimizer",
                "learning_rate", "weight_decay", "scheduler", "label_smoothing",
                "early_stop_patience", "pretrained_weights", "pretrained_sha256",
                "train_root", "val_root", "git_commit", "code_hash", "timestamp",
            }
            self.assertFalse(required - set(metadata))
            self.assertEqual(metadata["optimizer"], "AdamW")
            self.assertEqual(metadata["weight_decay"], 0.05)
            self.assertEqual(metadata["subset"], "R")


if __name__ == "__main__":
    unittest.main()
