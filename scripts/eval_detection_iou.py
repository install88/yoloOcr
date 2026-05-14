"""IoU-based real detection recall.

報表的 "Detection Recall 100%" 只表示 YOLO 每張圖至少畫了 1 個框，
這支腳本算「真實」recall: GT polygon 與 YOLO bbox 的 IoU >= 0.5 才算 hit。

Output:
  yoloOcr/results/det_iou_eval.txt
"""
import sys
import io
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from date_patterns import is_date_text
from config import DATASET_DIR, RESULTS_DIR, YOLO_MODEL

from ultralytics import YOLO

CONF_THRESH = 0.25
IOU_THRESH = 0.5


def poly_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main():
    split = "val"
    label_path = DATASET_DIR / f"{split}_label.txt"
    img_dir = DATASET_DIR / split

    model = YOLO(str(YOLO_MODEL))

    total_images = 0
    images_with_gt = 0
    images_with_any_hit = 0          # ≥1 GT box hit
    images_with_full_hit = 0         # all GT boxes hit
    images_with_zero_det = 0
    gt_total = 0
    gt_hit = 0
    pred_total = 0
    pred_matched = 0                 # YOLO box that matched some GT
    false_positives = 0

    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line in tqdm(lines, desc="eval det IoU"):
        if "\t" not in line:
            continue
        img_name, ann_json = line.split("\t", 1)
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
        try:
            anns = json.loads(ann_json)
        except json.JSONDecodeError:
            continue

        gt_bboxes = []
        for ann in anns:
            text = ann.get("transcription", "")
            if text == "###" or not is_date_text(text):
                continue
            gt_bboxes.append(poly_to_bbox(ann["points"]))

        total_images += 1
        if not gt_bboxes:
            continue
        images_with_gt += 1
        gt_total += len(gt_bboxes)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        result = model(img, conf=CONF_THRESH, verbose=False)[0]
        pred_bboxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes.xyxy.cpu().numpy():
                pred_bboxes.append(tuple(box))
        pred_total += len(pred_bboxes)

        if not pred_bboxes:
            images_with_zero_det += 1
            continue

        gt_was_hit = [False] * len(gt_bboxes)
        pred_was_matched = [False] * len(pred_bboxes)
        for gi, gb in enumerate(gt_bboxes):
            best_iou = 0.0
            best_pi = -1
            for pi, pb in enumerate(pred_bboxes):
                v = iou(gb, pb)
                if v > best_iou:
                    best_iou = v
                    best_pi = pi
            if best_iou >= IOU_THRESH and best_pi >= 0 and not pred_was_matched[best_pi]:
                gt_was_hit[gi] = True
                pred_was_matched[best_pi] = True

        n_hit = sum(gt_was_hit)
        gt_hit += n_hit
        pred_matched += sum(pred_was_matched)
        false_positives += sum(1 for m in pred_was_matched if not m)
        if n_hit > 0:
            images_with_any_hit += 1
        if n_hit == len(gt_bboxes):
            images_with_full_hit += 1

    # -------- summary --------
    out = []
    out.append("=" * 60)
    out.append(f"IoU-based Detection Evaluation  (IoU threshold = {IOU_THRESH})")
    out.append("=" * 60)
    out.append(f"Images total                : {total_images}")
    out.append(f"Images with date GT         : {images_with_gt}")
    out.append(f"GT boxes total              : {gt_total}")
    out.append(f"GT boxes hit                : {gt_hit}")
    out.append(f"Box-level Recall            : {gt_hit/gt_total:.1%}  ({gt_hit}/{gt_total})")
    out.append("")
    out.append(f"Images with ≥1 GT hit       : {images_with_any_hit}/{images_with_gt}  ({images_with_any_hit/images_with_gt:.1%})")
    out.append(f"Images with ALL GTs hit     : {images_with_full_hit}/{images_with_gt}  ({images_with_full_hit/images_with_gt:.1%})")
    out.append(f"Images with zero detection  : {images_with_zero_det}")
    out.append("")
    out.append(f"YOLO predictions total      : {pred_total}")
    out.append(f"YOLO predictions matched    : {pred_matched}")
    out.append(f"False positives (no GT)     : {false_positives}")
    if pred_total > 0:
        out.append(f"Precision                   : {pred_matched/pred_total:.1%}")

    summary = "\n".join(out)
    print(summary)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "det_iou_eval.txt").write_text(summary, encoding="utf-8")
    print(f"\n→ saved to {RESULTS_DIR / 'det_iou_eval.txt'}")


if __name__ == "__main__":
    main()
