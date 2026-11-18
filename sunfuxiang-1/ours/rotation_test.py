from ultralytics import YOLO
import yaml


model = YOLO(
    "/data/cle/sunfuxiang-1/ours/"
    "runs/detect/train68/weights/best.pt"
)

base_root = (
    "/data/cle/sunfuxiang-1/dataset/"
    "jptomato/test_rotation_hbb"
)

angles = [
    "angle_0",
    "angle_90",
    "angle_180",
    "angle_270",
]

results = {}

for angle in angles:

    print("\n" + "=" * 60)
    print(f"Testing: {angle}")
    print("=" * 60)

    rotation_root = (
        f"{base_root}/{angle}"
    )

    data_yaml = (
        f"{base_root}/{angle}.yaml"
    )

    data_config = {
        "path": rotation_root,
        "train": "images",
        "val": "images",
        "test": "images",
        "names": model.names,
    }

    with open(data_yaml, "w") as f:

        yaml.safe_dump(
            data_config,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    metrics = model.val(
        data=data_yaml,
        split="test",
        cfg="ultralytics/cfg/default.yaml",
        project="runs/rotation_hbb_test",
        name=angle
    )

    results[angle] = {
        "mAP50-95": metrics.box.map,
        "mAP50": metrics.box.map50,
        "mAP75": metrics.box.map75,
    }


print(
    "\n\n========== Final Results =========="
)

for angle, result in results.items():

    print(
        f"{angle:12s} | "
        f"mAP50-95="
        f"{result['mAP50-95']:.4f} | "
        f"mAP50="
        f"{result['mAP50']:.4f} | "
        f"mAP75="
        f"{result['mAP75']:.4f}"
    )