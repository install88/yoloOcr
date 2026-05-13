"""
Mine hard-negative rec crops for retraining.

Reads GT polygon annotations from dataset, crops each date region,
runs the current rec model, and saves every failing crop + its GT
label into rec_hard_negatives/ in PaddleOCR rec format.

Output:
  date-recognition/rec_hard_negatives/
    train/   val/     <- crop images
    train_label.txt   val_label.txt

Merge these into your rec_dataset before retraining:
  cat rec_hard_negatives/train_label.txt >> rec_dataset/train_label.txt
  cp  rec_hard_negatives/train/*  rec_dataset/train/
"""
import sys, io, json, re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
from date_patterns import is_date_text, extract_date

DATASET_DIR = Path("C:/Users/insta/Desktop/dataset")
MODELS_DIR  = Path("C:/Users/insta/Desktop/date-recognition/models")
OUT_DIR     = Path("C:/Users/insta/Desktop/date-recognition/rec_hard_negatives")

REC_MODEL   = MODELS_DIR / "rec_finetuned.onnx"
MIN_HEIGHT  = 48


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _ensure_min_height(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h < MIN_HEIGHT:
        scale = MIN_HEIGHT / h
        return cv2.resize(img, (max(1, int(w * scale)), MIN_HEIGHT),
                          interpolation=cv2.INTER_CUBIC)
    return img


def poly_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_region(img, points, padding=8):
    x1, y1, x2, y2 = poly_to_bbox(points)
    h, w = img.shape[:2]
    return img[max(0, y1 - padding):min(h, y2 + padding),
               max(0, x1 - padding):min(w, x2 + padding)]


def rec_text(ocr, crop):
    if crop.size == 0:
        return ""
    crop = _ensure_min_height(crop)
    try:
        result, _ = ocr(crop, use_det=False, use_cls=False, use_rec=True)
        if result:
            return " ".join(r[0] for r in result if r and isinstance(r[0], str))
    except Exception:
        pass
    return ""


def mine_split(split: str, label_path: Path, img_dir: Path, ocr,
               out_img_dir: Path, out_labels: list[str]):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    total = saved = 0

    with open(label_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc=split):
        tab_idx = line.index("\t")
        img_name = line[:tab_idx]
        annotations = json.loads(line[tab_idx + 1:])
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i, ann in enumerate(annotations):
            gt = ann.get("transcription", "")
            if gt == "###" or not is_date_text(gt):
                continue
            total += 1

            crop = crop_region(img, ann["points"])
            pred = rec_text(ocr, crop)
            pred_date = extract_date(pred)

            gt_dig  = _digits_only(extract_date(gt) or gt)
            pred_dig = _digits_only(pred_date or "")

            if gt_dig and pred_dig == gt_dig:
                continue  # correct — skip

            # Save failing crop
            stem = Path(img_name).stem
            crop_name = f"{stem}_box{i:02d}.jpg"
            dst = out_img_dir / crop_name
            crop_save = _ensure_min_height(crop)
            cv2.imwrite(str(dst), crop_save)
            out_labels.append(f"{crop_name}\t{gt}")
            saved += 1

    print(f"  [{split}] total={total}  failures_saved={saved}")


def main():
    from rapidocr_onnxruntime import RapidOCR
    if REC_MODEL.exists():
        print(f"Using fine-tuned rec: {REC_MODEL}")
        keys_path = MODELS_DIR / "ppocr_keys_v1.txt"
        ocr = RapidOCR(rec_model_path=str(REC_MODEL),
                       rec_keys_path=str(keys_path) if keys_path.exists() else None)
    else:
        print("Using default RapidOCR rec")
        ocr = RapidOCR()

    for split, label_file, img_sub in [
        ("train", "train_label.txt", "train"),
        ("val",   "val_label.txt",   "val"),
    ]:
        labels: list[str] = []
        mine_split(
            split,
            DATASET_DIR / label_file,
            DATASET_DIR / img_sub,
            ocr,
            OUT_DIR / split,
            labels,
        )
        lbl_path = OUT_DIR / f"{split}_label.txt"
        lbl_path.write_text("\n".join(labels), encoding="utf-8")
        print(f"  label file → {lbl_path}")

    print("\nDone. To merge into rec_dataset:")
    print("  cat rec_hard_negatives/train_label.txt >> rec_dataset/train_label.txt")
    print("  cp  rec_hard_negatives/train/*  rec_dataset/train/")
    print("  (same for val)")


if __name__ == "__main__":
    main()
