"""
Step 3: Convert PaddleOCR det annotations → YOLO format.
Only keeps regions where transcription matches a date pattern.
Output structure:
  date-recognition/yolo_dataset/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml
"""
import json
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
from date_patterns import is_date_text

DATASET_DIR  = Path("C:/Users/insta/Desktop/dataset")
OUTPUT_DIR   = Path("C:/Users/insta/Desktop/date-recognition/yolo_dataset")
RESULTS_DIR  = Path("C:/Users/insta/Desktop/date-recognition/results")

CLASS_NAMES = ["date"]


def poly_to_yolo(points: list, img_w: int, img_h: int) -> tuple[float, ...]:
    """Convert 4-corner polygon to YOLO (cx, cy, w, h) normalized."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def convert_split(split: str, label_path: Path, img_dir: Path):
    out_img_dir = OUTPUT_DIR / "images" / split
    out_lbl_dir = OUTPUT_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = 0

    with open(label_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc=f"convert {split}"):
        tab_idx = line.index("\t")
        img_name = line[:tab_idx]
        annotations = json.loads(line[tab_idx + 1:])
        src_path = img_dir / img_name
        if not src_path.exists():
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_lines: list[str] = []
        for ann in annotations:
            text = ann.get("transcription", "")
            if text == "###" or not is_date_text(text):
                continue
            cx, cy, bw, bh = poly_to_yolo(ann["points"], w, h)
            if bw <= 0 or bh <= 0:
                continue
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not yolo_lines:
            skipped += 1
            continue

        # copy image
        dst_img = out_img_dir / img_name
        shutil.copy2(src_path, dst_img)

        # write label
        lbl_name = Path(img_name).stem + ".txt"
        (out_lbl_dir / lbl_name).write_text("\n".join(yolo_lines), encoding="utf-8")
        kept += 1

    print(f"  [{split}] kept={kept}  skipped(no date)={skipped}")
    return kept


def write_yaml(train_count: int, val_count: int):
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# Stats: train={train_count}, val={val_count}
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"  data.yaml → {yaml_path}")


def main():
    print("Converting PaddleOCR annotations → YOLO format...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tc = convert_split("train", DATASET_DIR / "train_label.txt", DATASET_DIR / "train")
    vc = convert_split("val",   DATASET_DIR / "val_label.txt",   DATASET_DIR / "val")
    write_yaml(tc, vc)
    print("\nConversion complete.")


if __name__ == "__main__":
    main()
