#!/usr/bin/env python3
"""CPU-only validation of Full and manifest-filtered datasets; never trains a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

import torch
from torchvision import transforms


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_manifest import filter_training_samples, summarize_training_samples, sha256_file  # noqa: E402
from my_dataset import MyDataSet  # noqa: E402
from train_cle_fair import prepare_training_samples  # noqa: E402


EXPECTED_COUNTS = {"R": 3500, "V": 3474, "C": 3749, "RV": 3103, "RC": 3141, "VC": 3115}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(root: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in root.rglob("*")
    )


def ordered_path_label_sha256(paths, labels, root: Path) -> str:
    digest = hashlib.sha256()
    for path, label in zip(paths, labels):
        relative = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(f"{relative}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def main(args: argparse.Namespace) -> int:
    train_root = args.train_root.resolve(strict=True)
    val_root = args.val_root.resolve(strict=True)
    test_root = args.test_root.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="cle_manifest_validate_") as temporary:
        original_cwd = Path.cwd()
        try:
            os.chdir(temporary)
            full_paths, full_labels, manifest_info = prepare_training_samples(
                str(train_root), train_manifest="", data_path_val=str(val_root)
            )
        finally:
            os.chdir(original_cwd)
    if manifest_info is not None:
        raise ValueError("no-manifest Full scan unexpectedly returned manifest metadata")

    full_info = summarize_training_samples(full_paths, full_labels, train_root)
    full_info["ordered_path_label_sha256"] = ordered_path_label_sha256(
        full_paths, full_labels, train_root
    )
    expected_full = {
        "actual_train_count": 4977,
        "normal_count": 1578,
        "benign_count": 2136,
        "malignant_count": 1263,
    }
    for key, expected in expected_full.items():
        if full_info[key] != expected:
            raise ValueError(f"Full {key}: expected {expected}, got {full_info[key]}")
    if (
        args.baseline_ordered_sha256
        and full_info["ordered_path_label_sha256"] != args.baseline_ordered_sha256
    ):
        raise ValueError("Full ordered path/label hash changed from pre-modification baseline")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )
    report = {
        "device": "cpu",
        "training_started": False,
        "full": full_info,
        "fixed_val_count": count_images(val_root),
        "fixed_test_count": count_images(test_root),
        "pretrained_sha256": sha256_file(args.pretrained),
        "subsets": {},
    }
    if report["fixed_val_count"] != 538 or report["fixed_test_count"] != 671:
        raise ValueError("fixed val/test counts changed")
    if report["pretrained_sha256"] != args.expected_pretrained_sha256:
        raise ValueError("pretrained ResNet34 SHA-256 mismatch")

    for subset, expected_count in EXPECTED_COUNTS.items():
        manifest = args.manifest_dir / f"{subset}.csv"
        paths, labels, info = filter_training_samples(
            full_paths,
            full_labels,
            train_root,
            manifest,
            exclusion_roots=[val_root, test_root],
        )
        if info["actual_train_count"] != expected_count:
            raise ValueError(
                f"{subset}: expected {expected_count}, got {info['actual_train_count']}"
            )
        dataset = MyDataSet(paths, labels, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        images, batch_labels = next(iter(loader))
        expected_shape = [min(args.batch_size, expected_count), 3, 224, 224]
        if list(images.shape) != expected_shape:
            raise ValueError(
                f"{subset}: batch shape {list(images.shape)} != {expected_shape}"
            )
        if list(batch_labels.shape) != [expected_shape[0]]:
            raise ValueError(f"{subset}: label shape {list(batch_labels.shape)}")
        report["subsets"][subset] = {
            **info,
            "images_shape": list(images.shape),
            "labels_shape": list(batch_labels.shape),
            "labels_dtype": str(batch_labels.dtype),
        }

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parent = REPO.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=parent / "datasets/3class/train")
    parser.add_argument("--val-root", type=Path, default=parent / "datasets/3class/val")
    parser.add_argument("--test-root", type=Path, default=parent / "datasets/3class/test")
    parser.add_argument("--manifest-dir", type=Path, default=REPO / "manifests/consensus_ablation")
    parser.add_argument("--pretrained", type=Path, default=REPO / "pth/resnet34-333f7ec4.pth")
    parser.add_argument(
        "--expected-pretrained-sha256",
        default="333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef",
    )
    parser.add_argument("--baseline-ordered-sha256", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


if __name__ == "__main__":
    raise SystemExit(main(make_parser().parse_args()))
