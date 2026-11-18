from pathlib import Path
import shutil

import cv2
from tqdm import tqdm


# ============================================================
# 1. Configuration
# ============================================================

# 原始测试集
SOURCE_ROOT = Path(
    "/data/cle/sunfuxiang-1/dataset/jptomato/test"
)

# 新生成的数据集
# 不再使用旧 test_rotation，避免旧错误数据和 cache 混入
OUTPUT_ROOT = Path(
    "/data/cle/sunfuxiang-1/dataset/jptomato/test_rotation_hbb"
)

# 正角度定义为逆时针旋转
ANGLES = [0, 90, 180, 270]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# 是否先清空已经存在的 angle_xxx 目录
# 建议保持 True，防止旧文件残留
CLEAR_OLD_OUTPUT = True


# ============================================================
# 2. Dataset structure
# ============================================================

def find_dataset_dirs(root: Path):
    """
    支持标准 YOLO 数据集结构：

    test/
    ├── images/
    └── labels/
    """

    image_dir = root / "images"
    label_dir = root / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}"
        )

    return image_dir, label_dir


# ============================================================
# 3. Angle directory name
# ============================================================

def angle_name(angle: int) -> str:
    return f"angle_{angle}"


# ============================================================
# 4. Rotate image
# ============================================================

def rotate_image(image, angle: int):
    """
    正角度定义为逆时针。

    0   : 不旋转
    90  : 逆时针 90°
    180 : 180°
    270 : 逆时针 270°，等价于顺时针 90°
    """

    if angle == 0:
        return image.copy()

    elif angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    elif angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    elif angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    else:
        raise ValueError(
            f"Unsupported angle: {angle}"
        )


# ============================================================
# 5. Strict YOLO HBB transformation
# ============================================================

def rotate_yolo_hbb(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    angle: int
):
    """
    输入和输出均为 YOLO normalized xywh：

        xc, yc, w, h

    由于旋转角度仅为：
        0 / 90 / 180 / 270

    axis-aligned bounding box 旋转之后依旧是
    axis-aligned bounding box。

    因此不存在 ±15°、±30° 时的外接框膨胀问题。
    """

    if angle == 0:

        new_xc = xc
        new_yc = yc
        new_w = bw
        new_h = bh

    elif angle == 90:
        # Counter-clockwise 90°
        #
        # x' = y
        # y' = 1 - x
        #
        # width / height swap

        new_xc = yc
        new_yc = 1.0 - xc

        new_w = bh
        new_h = bw

    elif angle == 180:

        new_xc = 1.0 - xc
        new_yc = 1.0 - yc

        new_w = bw
        new_h = bh

    elif angle == 270:
        # Counter-clockwise 270°
        # == clockwise 90°
        #
        # x' = 1 - y
        # y' = x

        new_xc = 1.0 - yc
        new_yc = xc

        new_w = bh
        new_h = bw

    else:
        raise ValueError(
            f"Unsupported angle: {angle}"
        )

    # 防止极小浮点误差
    new_xc = max(
        0.0,
        min(1.0, new_xc)
    )

    new_yc = max(
        0.0,
        min(1.0, new_yc)
    )

    new_w = max(
        0.0,
        min(1.0, new_w)
    )

    new_h = max(
        0.0,
        min(1.0, new_h)
    )

    return (
        new_xc,
        new_yc,
        new_w,
        new_h
    )


# ============================================================
# 6. Transform one label file
# ============================================================

def transform_label_file(
    input_label: Path,
    output_label: Path,
    angle: int
):
    """
    读取：

        class xc yc w h

    并生成旋转后的 YOLO HBB 标签。
    """

    output_label.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 没有标签文件时，生成空标签
    if not input_label.exists():

        output_label.write_text("")

        return 0

    text = input_label.read_text().strip()

    if not text:

        output_label.write_text("")

        return 0

    transformed_lines = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        parts = line.strip().split()

        # 这里只接受 YOLO Detect HBB
        if len(parts) != 5:

            raise ValueError(
                "\nUnsupported label format.\n"
                f"File: {input_label}\n"
                f"Line: {line_number}\n"
                f"Content: {line}\n\n"
                "Expected format:\n"
                "class x_center y_center width height"
            )

        class_id = parts[0]

        xc, yc, bw, bh = map(
            float,
            parts[1:]
        )

        # 检查原始标签是否合法
        values = [xc, yc, bw, bh]

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"Invalid normalized bbox in "
                f"{input_label}:\n{line}"
            )

        (
            new_xc,
            new_yc,
            new_w,
            new_h
        ) = rotate_yolo_hbb(
            xc,
            yc,
            bw,
            bh,
            angle
        )

        transformed_lines.append(
            f"{class_id} "
            f"{new_xc:.8f} "
            f"{new_yc:.8f} "
            f"{new_w:.8f} "
            f"{new_h:.8f}"
        )

    output_label.write_text(
        "\n".join(transformed_lines)
        + "\n"
    )

    return len(transformed_lines)


# ============================================================
# 7. Save image
# ============================================================

def save_image(
    output_path: Path,
    image
):
    """
    所有角度，包括 0°，都通过同一个保存流程重新写入，
    避免只有旋转组发生重新编码而 0° 直接复制所产生的
    不一致。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ext = output_path.suffix.lower()

    # JPEG 用较高质量保存
    if ext in {".jpg", ".jpeg"}:

        success = cv2.imwrite(
            str(output_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                100
            ]
        )

    elif ext == ".png":

        success = cv2.imwrite(
            str(output_path),
            image,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                3
            ]
        )

    else:

        success = cv2.imwrite(
            str(output_path),
            image
        )

    if not success:
        raise IOError(
            f"Failed to save image: "
            f"{output_path}"
        )


# ============================================================
# 8. Main
# ============================================================

def main():

    image_dir, label_dir = find_dataset_dirs(
        SOURCE_ROOT
    )

    print("=" * 70)
    print("Strict HBB Rotation Test Set Generator")
    print("=" * 70)

    print(f"Source images : {image_dir}")
    print(f"Source labels : {label_dir}")
    print(f"Output root   : {OUTPUT_ROOT}")
    print(f"Angles        : {ANGLES}")
    print(
        "Convention    : positive angle = "
        "counter-clockwise"
    )

    # --------------------------------------------------------
    # Find all images
    # --------------------------------------------------------

    image_paths = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if not image_paths:

        raise RuntimeError(
            f"No images found in: {image_dir}"
        )

    print(
        f"Number of images: {len(image_paths)}"
    )

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    for angle in ANGLES:

        root = (
            OUTPUT_ROOT /
            angle_name(angle)
        )

        if (
            CLEAR_OLD_OUTPUT
            and root.exists()
        ):

            print(
                f"Removing old directory: "
                f"{root}"
            )

            shutil.rmtree(root)

        (
            root /
            "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            root /
            "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    box_counts = {
        angle: 0
        for angle in ANGLES
    }

    image_counts = {
        angle: 0
        for angle in ANGLES
    }

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for image_path in tqdm(
        image_paths,
        desc="Generating strict HBB sets"
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"\nWarning: failed to read "
                f"{image_path}"
            )

            continue

        relative_image_path = (
            image_path.relative_to(
                image_dir
            )
        )

        relative_label_path = (
            relative_image_path
            .with_suffix(".txt")
        )

        source_label_path = (
            label_dir /
            relative_label_path
        )

        for angle in ANGLES:

            root = (
                OUTPUT_ROOT /
                angle_name(angle)
            )

            output_image_path = (
                root /
                "images" /
                relative_image_path
            )

            output_label_path = (
                root /
                "labels" /
                relative_label_path
            )

            # --------------------------------------------
            # Image
            # --------------------------------------------

            rotated_image = rotate_image(
                image,
                angle
            )

            save_image(
                output_image_path,
                rotated_image
            )

            # --------------------------------------------
            # Label
            # --------------------------------------------

            num_boxes = transform_label_file(
                source_label_path,
                output_label_path,
                angle
            )

            image_counts[angle] += 1
            box_counts[angle] += num_boxes

    # ========================================================
    # 9. Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("Generation completed")
    print("=" * 70)

    for angle in ANGLES:

        print(
            f"{angle:3d}° | "
            f"images = "
            f"{image_counts[angle]:4d} | "
            f"boxes = "
            f"{box_counts[angle]:5d}"
        )

    # 严格 HBB 旋转的一个基本 sanity check：
    # 所有角度 bbox 数量必须完全一致。

    counts = list(
        box_counts.values()
    )

    if len(set(counts)) == 1:

        print(
            "\n[PASS] All rotation sets have "
            "the same number of bounding boxes."
        )

    else:

        print(
            "\n[WARNING] Bounding box counts "
            "are inconsistent!"
        )

    print("\nOutput directories:")

    for angle in ANGLES:

        print(
            OUTPUT_ROOT /
            angle_name(angle)
        )


if __name__ == "__main__":
    main()