"""
Step 2: Baseline recognition test.
  - Crops date regions from train/val images using ground-truth polygons
  - Runs RapidOCR (PPOCRv4 via ONNX) recognition on each crop
  - Reports character-level accuracy and common failure cases
"""
import sys
import io
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rapidocr_onnxruntime import RapidOCR

# Force UTF-8 stdout so Chinese chars don't crash on Windows CP950
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
from date_patterns import is_date_text

DATASET_DIR = Path("C:/Users/insta/Desktop/dataset")
RESULTS_DIR = Path("C:/Users/insta/Desktop/date-recognition/results")
RESULTS_DIR.mkdir(exist_ok=True)


def poly_to_bbox(points: list) -> tuple[int, int, int, int]:
    """Convert 4-point polygon to axis-aligned bounding box."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_region(img: np.ndarray, points: list, padding: int = 8) -> np.ndarray:
    x1, y1, x2, y2 = poly_to_bbox(points)
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img[y1:y2, x1:x2]


def cer(gt: str, pred: str) -> float:
    """Character Error Rate (simple edit distance / len(gt))."""
    gt_chars  = list(gt.replace(" ", ""))
    pred_chars = list(pred.replace(" ", ""))
    if not gt_chars:
        return 0.0
    m, n = len(gt_chars), len(pred_chars)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if gt_chars[i - 1] == pred_chars[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / m


def run_split(split: str, label_path: Path, img_dir: Path, ocr: RapidOCR):
    total = correct = 0
    cer_sum = 0.0
    failures: list[dict] = []

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

        for ann in annotations:
            gt_text = ann.get("transcription", "")
            if gt_text == "###" or not is_date_text(gt_text):
                continue

            crop = crop_region(img, ann["points"])
            if crop.size == 0:
                continue

            try:
                # use_det=False: rec-only returns [text, score], not [bbox, text, score]
                result, _ = ocr(crop, use_det=False, use_cls=False, use_rec=True)
                pred_text = ""
                if result:
                    pred_text = " ".join(r[0] for r in result if r and isinstance(r[0], str))
            except Exception:
                pred_text = ""

            total += 1
            err = cer(gt_text, pred_text)
            cer_sum += err
            if err == 0.0:
                correct += 1
            elif len(failures) < 30:
                failures.append({
                    "img": img_name,
                    "gt": gt_text,
                    "pred": pred_text,
                    "cer": round(err, 3),
                })

    avg_cer = cer_sum / total if total else 0
    acc = correct / total if total else 0
    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "avg_cer": avg_cer,
        "failures": failures,
    }


def main():
    print("Initializing RapidOCR (PPOCRv4 mobile)...")
    ocr = RapidOCR()

    report_lines: list[str] = []

    for split, label_file, img_subdir in [
        ("Train", "train_label.txt", "train"),
        ("Val",   "val_label.txt",   "val"),
    ]:
        r = run_split(
            split,
            DATASET_DIR / label_file,
            DATASET_DIR / img_subdir,
            ocr,
        )
        header = f"\n[{split}] total={r['total']}  exact={r['correct']}  accuracy={r['accuracy']:.1%}  avg_CER={r['avg_cer']:.3f}"
        print(header)
        report_lines.append(header)
        print("  Top failures (CER > 0):")
        for f in sorted(r["failures"], key=lambda x: -x["cer"])[:10]:
            line = f"    CER={f['cer']:.2f}  GT='{f['gt']}'  PRED='{f['pred']}'  ({f['img']})"
            print(line)
            report_lines.append(line)

    out_path = RESULTS_DIR / "baseline_report.txt"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
