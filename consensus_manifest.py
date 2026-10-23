"""Validation and filtering helpers for consensus-ablation train manifests."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


MANIFEST_FIELDS = ("sample_id", "class_name", "relative_path")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_TO_COUNT_KEY = {
    "body_of_stomach_normal": "normal_count",
    "Benign_lesions": "benign_count",
    "Malignant_lesions": "malignant_count",
}
SAMPLE_ID_RE = re.compile(r"(S_[0-9A-Za-z]+)(?=\.[^.]+$)")


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_sample_id(path: os.PathLike[str] | str) -> str:
    match = SAMPLE_ID_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"cannot extract sample_id from filename: {Path(path).name}")
    return match.group(1)


def _safe_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"relative_path is outside train root: {value!r}")
    return relative


def _resolved_inside(root: Path, relative: PurePosixPath) -> Path:
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"relative_path is outside train root: {relative}") from exc
    return candidate


def scan_sample_ids(roots: Iterable[os.PathLike[str] | str]) -> set[str]:
    sample_ids: set[str] = set()
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                sample_id = extract_sample_id(path)
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample_id across exclusion roots: {sample_id}")
                sample_ids.add(sample_id)
    return sample_ids


def summarize_training_samples(
    paths: Sequence[str], labels: Sequence[int], train_root: os.PathLike[str] | str
) -> dict:
    if len(paths) != len(labels):
        raise ValueError("training paths and labels have different lengths")
    root = Path(train_root).resolve(strict=True)
    counts = Counter()
    for path_value in paths:
        path = Path(path_value).resolve(strict=True)
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"training path is outside train root: {path}") from exc
        class_name = relative.parts[0]
        if class_name not in CLASS_TO_COUNT_KEY:
            raise ValueError(f"unknown class directory: {class_name}")
        counts[CLASS_TO_COUNT_KEY[class_name]] += 1
    return {
        "actual_train_count": len(paths),
        "normal_count": counts["normal_count"],
        "benign_count": counts["benign_count"],
        "malignant_count": counts["malignant_count"],
    }


def filter_training_samples(
    train_paths: Sequence[str],
    train_labels: Sequence[int],
    train_root: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    exclusion_roots: Iterable[os.PathLike[str] | str] = (),
) -> tuple[list[str], list[int], dict]:
    """Validate a manifest and filter an already-scanned Full train list.

    Returned paths retain the original scanner order, so the no-manifest branch and
    the existing class-index construction remain untouched.
    """
    if len(train_paths) != len(train_labels):
        raise ValueError("training paths and labels have different lengths")

    root = Path(train_root).resolve(strict=True)
    manifest = Path(manifest_path).resolve(strict=True)

    by_relative: dict[str, tuple[str, int, str]] = {}
    by_sample_id: dict[str, str] = {}
    for path_value, label in zip(train_paths, train_labels):
        resolved = Path(path_value).resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"training path is outside train root: {resolved}") from exc
        sample_id = extract_sample_id(resolved)
        class_name = PurePosixPath(relative).parts[0]
        if relative in by_relative:
            raise ValueError(f"duplicate relative path in Full train: {relative}")
        if sample_id in by_sample_id:
            raise ValueError(f"duplicate sample_id in Full train: {sample_id}")
        by_relative[relative] = (str(resolved), int(label), sample_id)
        by_sample_id[sample_id] = relative

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(
                f"manifest header must be {','.join(MANIFEST_FIELDS)}; got {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("manifest is empty")

    selected_relatives: set[str] = set()
    selected_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        class_name = row["class_name"].strip()
        relative = _safe_relative_path(row["relative_path"].strip())
        relative_text = relative.as_posix()

        if sample_id in selected_ids:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        if relative_text in selected_relatives:
            raise ValueError(f"duplicate relative_path in manifest: {relative_text}")
        selected_ids.add(sample_id)
        selected_relatives.add(relative_text)

        resolved = _resolved_inside(root, relative)
        if relative.parts[0] != class_name:
            raise ValueError(
                f"manifest line {line_number}: class_name {class_name!r} does not match "
                f"relative_path parent {relative.parts[0]!r}"
            )
        if relative_text not in by_relative:
            raise ValueError(f"unknown relative_path in manifest: {relative_text}")
        _, _, indexed_sample_id = by_relative[relative_text]
        if sample_id not in by_sample_id:
            raise ValueError(f"unknown sample_id in manifest: {sample_id}")
        if indexed_sample_id != sample_id or extract_sample_id(resolved) != sample_id:
            raise ValueError(
                f"manifest line {line_number}: sample_id/path mismatch for {sample_id}"
            )
        if by_sample_id[sample_id] != relative_text:
            raise ValueError(
                f"manifest line {line_number}: sample_id maps to a different Full path"
            )

    excluded = scan_sample_ids(exclusion_roots)
    overlap = sorted(selected_ids & excluded)
    if overlap:
        raise ValueError(
            f"manifest overlaps validation/test sample_ids ({len(overlap)}), first={overlap[0]}"
        )

    filtered_paths: list[str] = []
    filtered_labels: list[int] = []
    for path_value, label in zip(train_paths, train_labels):
        relative = Path(path_value).resolve(strict=True).relative_to(root).as_posix()
        if relative in selected_relatives:
            filtered_paths.append(path_value)
            filtered_labels.append(label)

    if len(filtered_paths) != len(rows):
        raise ValueError(
            f"actual filtered count {len(filtered_paths)} != manifest count {len(rows)}"
        )

    info = summarize_training_samples(filtered_paths, filtered_labels, root)
    info.update(
        {
            "subset": manifest.stem,
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        }
    )
    return filtered_paths, filtered_labels, info
