#!/usr/bin/env python3
"""Build and validate R/V/C/RV/RC/VC manifests without copying images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from consensus_manifest import (  # noqa: E402
    CLASS_TO_COUNT_KEY,
    MANIFEST_FIELDS,
    SUPPORTED_EXTENSIONS,
    extract_sample_id,
    sha256_file,
)


EXPECTED_DEFAULT = {
    "R": 3500,
    "V": 3474,
    "C": 3749,
    "RV": 3103,
    "RC": 3141,
    "VC": 3115,
    "RVC": 2880,
}
ALLOWED_GRADES = {"A", "B", "C", "D", "E"}


def read_unique_rows(path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_number}: empty sample_id")
            if sample_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            rows[sample_id] = row
    return rows


def scan_split(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        sample_id = extract_sample_id(path)
        if sample_id in result:
            raise ValueError(f"{root}: duplicate sample_id {sample_id}")
        result[sample_id] = path
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o664)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def render_manifest(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def parse_expected(values: list[str]) -> dict[str, int]:
    expected = dict(EXPECTED_DEFAULT)
    for value in values:
        name, separator, raw_count = value.partition("=")
        if not separator or name not in expected:
            raise ValueError(f"invalid --expected value: {value}")
        expected[name] = int(raw_count)
    return expected


def build(args: argparse.Namespace) -> dict:
    full_train = args.full_train.resolve(strict=True)
    hq_train = args.hq_train.resolve(strict=True)
    fixed_val = (args.fixed_val or full_train.parent / "val").resolve(strict=True)
    fixed_test = (args.fixed_test or full_train.parent / "test").resolve(strict=True)
    expected = parse_expected(args.expected)

    r_rows = read_unique_rows(args.resnet_csv, {"sample_id", "grade", "image_path"})
    v_rows = read_unique_rows(args.vit_csv, {"sample_id", "grade", "image_path"})
    c_rows = read_unique_rows(
        args.cufit_csv,
        {"sample_id", "cufit_lpm_correct_rate", "image_path"},
    )
    authoritative_rows = read_unique_rows(args.authoritative_rvc, {"sample_id"})

    universes = {"R": set(r_rows), "V": set(v_rows), "C": set(c_rows)}
    if not (universes["R"] == universes["V"] == universes["C"]):
        raise ValueError("R/V/C input sample_id universes differ")

    full_index = scan_split(full_train)
    val_ids = set(scan_split(fixed_val))
    test_ids = set(scan_split(fixed_test))
    if set(full_index) != universes["R"]:
        raise ValueError(
            f"Full train and CSV universes differ: full-only={len(set(full_index)-universes['R'])}, "
            f"csv-only={len(universes['R']-set(full_index))}"
        )
    if set(full_index) & (val_ids | test_ids):
        raise ValueError("Full train sample_ids overlap fixed val/test")

    for source_name, rows in (("ResNet", r_rows), ("ViT", v_rows), ("CUFIT", c_rows)):
        for sample_id, row in rows.items():
            recorded = Path(row["image_path"])
            actual = full_index[sample_id]
            if recorded.name != actual.name or recorded.parent.name != actual.parent.name:
                raise ValueError(
                    f"{source_name}: image_path mismatch for {sample_id}: "
                    f"recorded={recorded}, actual={actual}"
                )

    for source_name, rows in (("ResNet", r_rows), ("ViT", v_rows)):
        bad = sorted({row["grade"].strip() for row in rows.values()} - ALLOWED_GRADES)
        if bad:
            raise ValueError(f"{source_name}: unknown grade values {bad}")
    rates: dict[str, float] = {}
    for sample_id, row in c_rows.items():
        try:
            rate = float(row["cufit_lpm_correct_rate"])
        except ValueError as exc:
            raise ValueError(f"CUFIT: invalid correct_rate for {sample_id}") from exc
        if not math.isfinite(rate):
            raise ValueError(f"CUFIT: non-finite correct_rate for {sample_id}")
        rates[sample_id] = rate

    good_r = {sid for sid, row in r_rows.items() if row["grade"].strip() in {"A", "B", "C"}}
    good_v = {sid for sid, row in v_rows.items() if row["grade"].strip() in {"A", "B", "C"}}
    good_c = {sid for sid, rate in rates.items() if rate >= 0.5}
    subsets = {
        "R": good_r,
        "V": good_v,
        "C": good_c,
        "RV": good_r & good_v,
        "RC": good_r & good_c,
        "VC": good_v & good_c,
    }
    computed_rvc = good_r & good_v & good_c
    authoritative_rvc = set(authoritative_rows)
    hq_index = scan_split(hq_train)
    hq_ids = set(hq_index)

    actual_counts = {name: len(values) for name, values in subsets.items()}
    actual_counts["RVC"] = len(computed_rvc)
    if actual_counts != expected:
        raise ValueError(f"subset counts differ: actual={actual_counts}, expected={expected}")
    if computed_rvc != authoritative_rvc or computed_rvc != hq_ids:
        raise ValueError(
            "RVC mismatch: "
            f"computed-authoritative={len(computed_rvc-authoritative_rvc)}, "
            f"authoritative-computed={len(authoritative_rvc-computed_rvc)}, "
            f"computed-hq={len(computed_rvc-hq_ids)}, hq-computed={len(hq_ids-computed_rvc)}"
        )
    for sample_id in computed_rvc:
        if hq_index[sample_id].parent.name != full_index[sample_id].parent.name:
            raise ValueError(f"HQ class mismatch for {sample_id}")

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "full_train": str(full_train),
        "fixed_val": {"path": str(fixed_val), "count": len(val_ids)},
        "fixed_test": {"path": str(fixed_test), "count": len(test_ids)},
        "inputs": {
            "resnet_csv": {"path": str(args.resnet_csv.resolve()), "sha256": sha256_file(args.resnet_csv)},
            "vit_csv": {"path": str(args.vit_csv.resolve()), "sha256": sha256_file(args.vit_csv)},
            "cufit_csv": {"path": str(args.cufit_csv.resolve()), "sha256": sha256_file(args.cufit_csv)},
            "authoritative_rvc": {
                "path": str(args.authoritative_rvc.resolve()),
                "sha256": sha256_file(args.authoritative_rvc),
            },
        },
        "subsets": {},
        "rvc_validation": {
            "computed_count": len(computed_rvc),
            "authoritative_count": len(authoritative_rvc),
            "hq_count": len(hq_ids),
            "computed_minus_authoritative": 0,
            "authoritative_minus_computed": 0,
            "computed_minus_hq": 0,
            "hq_minus_computed": 0,
            "training_manifest_generated": False,
        },
    }

    for name, sample_ids in subsets.items():
        manifest_rows = []
        class_counts = Counter()
        for sample_id in sorted(sample_ids):
            path = full_index[sample_id]
            relative = path.relative_to(full_train)
            class_name = relative.parts[0]
            if class_name not in CLASS_TO_COUNT_KEY:
                raise ValueError(f"unknown class directory: {class_name}")
            if extract_sample_id(path) != sample_id:
                raise ValueError(f"sample_id/path mismatch: {sample_id} -> {path}")
            class_counts[CLASS_TO_COUNT_KEY[class_name]] += 1
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "relative_path": relative.as_posix(),
                }
            )
        rendered = render_manifest(manifest_rows)
        target = args.out_dir / f"{name}.csv"
        if args.verify_only:
            if not target.is_file():
                raise FileNotFoundError(f"missing manifest: {target}")
            existing = target.read_text(encoding="utf-8")
            if existing != rendered:
                raise ValueError(f"manifest content mismatch: {target}")
        else:
            if target.exists() and target.read_text(encoding="utf-8") != rendered:
                raise FileExistsError(f"refusing to overwrite different manifest: {target}")
            atomic_write_text(target, rendered)
        metadata["subsets"][name] = {
            "path": str(target.resolve()),
            "total": len(sample_ids),
            "normal": class_counts["normal_count"],
            "benign": class_counts["benign_count"],
            "malignant": class_counts["malignant_count"],
            "sha256": sha256_file(target),
        }

    if not args.verify_only:
        atomic_write_text(
            args.out_dir / "MANIFEST_METADATA.json",
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True))
    return metadata


def make_parser() -> argparse.ArgumentParser:
    parent = REPO.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-train", type=Path, default=parent / "datasets/3class/train")
    parser.add_argument("--resnet-csv", type=Path, default=REPO / "outputs_3class_resnet_newnormal/train_sample_grading.csv")
    parser.add_argument("--vit-csv", type=Path, default=REPO / "outputs_3class_vit_newnormal/train_sample_grading.csv")
    parser.add_argument("--cufit-csv", type=Path, default=parent / "CUFIT0613/cufit_lpm_oof_3class_newnormal.csv")
    parser.add_argument("--authoritative-rvc", type=Path, default=parent / "CUFIT0613/consensus3_oof_good_sample_ids.csv")
    parser.add_argument("--hq-train", type=Path, default=parent / "datasets/3class_highquality/train")
    parser.add_argument("--fixed-val", type=Path, default=None)
    parser.add_argument("--fixed-test", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=REPO / "manifests/consensus_ablation")
    parser.add_argument("--expected", nargs="*", default=[])
    parser.add_argument("--verify-only", action="store_true")
    return parser


if __name__ == "__main__":
    build(make_parser().parse_args())
