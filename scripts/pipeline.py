"""
End-to-end date recognition pipeline.
  YOLO det (best.pt)  →  crop + padding  →  RapidOCR rec-only  →  regex extract_date()

Usage:
  # single image
  python pipeline.py path/to/image.jpg

  # batch eval on val set (reports end-to-end accuracy)
  python pipeline.py --eval-val

  # save annotated output image
  python pipeline.py path/to/image.jpg --save
"""
import sys
import io
import re
import json
import base64
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from date_patterns import extract_date, is_date_text
from config import DATASET_DIR, MODELS_DIR, RESULTS_DIR, YOLO_MODEL, REC_MODEL

from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESH = 0.25
PADDING     = 12          # px added around each detected bbox before rec
REPORT_HTML = RESULTS_DIR / "report.html"


# ── helpers ──────────────────────────────────────────────────────────────────

def _digits_only(s: str) -> str:
    """Strip everything except digits — used for separator-agnostic date comparison."""
    return re.sub(r"\D", "", s)


def _img_to_b64(img: np.ndarray, max_width: int = 500) -> str:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


def _draw_report_boxes(img: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = (34, 180, 34) if d["date"] else (34, 100, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (d["date"] or d["rec"] or "?")[:30]
        # ASCII/digits only for cv2 text (Chinese chars not supported)
        label_safe = re.sub(r"[^\x00-\x7F]", "", label) or "?"
        (tw, th), _ = cv2.getTextSize(label_safe, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 6)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(out, label_safe, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def _generate_html_report(records: list[dict], stats: dict, out_path: Path):
    pass_n  = sum(1 for r in records if r["matched"])
    fail_n  = len(records) - pass_n
    nodet_n = sum(1 for r in records if r["no_det"])

    # Failures first, then passes — within each group sorted by filename
    records = (sorted((r for r in records if not r["matched"]), key=lambda r: r["img"]) +
               sorted((r for r in records if r["matched"]),     key=lambda r: r["img"]))

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    cards = []
    for rec in records:
        status  = "pass" if rec["matched"] else "fail"
        badge   = ('<span class="badge pass-badge">✓ 正確</span>' if rec["matched"]
                   else '<span class="badge fail-badge">✗ 錯誤</span>')
        img_tag = (f'<img src="data:image/jpeg;base64,{rec["b64"]}">'
                   if rec["b64"] else '<div class="no-img">（圖片無法載入）</div>')
        gt_str   = esc(", ".join(rec["gt_dates"]))
        pred_str = esc(", ".join(rec["pred_dates"])) if rec["pred_dates"] else "<em>無</em>"
        rec_str  = esc(", ".join(rec["rec_texts"]))  if rec["rec_texts"]  else "<em>無偵測</em>"
        cards.append(f"""
<div class="card {status}" data-status="{status}">
  <div class="card-hdr">{badge} <span class="fname">{esc(rec['img'])}</span></div>
  {img_tag}
  <table>
    <tr><td class="lbl">GT</td><td>{gt_str}</td></tr>
    <tr><td class="lbl">DATE</td><td>{pred_str}</td></tr>
    <tr><td class="lbl">REC</td><td>{rec_str}</td></tr>
  </table>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>Date Recognition Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:Arial,sans-serif;background:#f0f2f5;padding:20px;color:#333}}
  h1{{margin-bottom:16px;font-size:1.4em}}
  .stats{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}}
  .stat{{background:white;border-radius:8px;padding:14px 20px;text-align:center;
         box-shadow:0 1px 4px rgba(0,0,0,.1);min-width:110px}}
  .stat .val{{font-size:1.8em;font-weight:700}}
  .stat .lbl2{{font-size:.78em;color:#888;margin-top:2px}}
  .blue{{color:#2980b9}}.green{{color:#27ae60}}.red{{color:#e74c3c}}.grey{{color:#555}}
  .filters{{margin-bottom:14px;display:flex;gap:8px;align-items:center}}
  .filters span{{font-size:.9em;color:#666}}
  button{{padding:7px 14px;border:none;border-radius:5px;cursor:pointer;font-size:.85em;font-weight:600}}
  .btn-all{{background:#3498db;color:#fff}} .btn-fail{{background:#e74c3c;color:#fff}}
  .btn-pass{{background:#27ae60;color:#fff}}
  .grid{{display:flex;flex-wrap:wrap;gap:14px}}
  .card{{background:white;border-radius:8px;padding:12px;width:380px;
         box-shadow:0 1px 4px rgba(0,0,0,.1);border-left:5px solid #ccc}}
  .card.fail{{border-left-color:#e74c3c}} .card.pass{{border-left-color:#27ae60}}
  .card-hdr{{font-size:.72em;color:#666;margin-bottom:8px;display:flex;gap:6px;align-items:center}}
  .fname{{word-break:break-all}}
  .card img{{width:100%;border-radius:4px;margin-bottom:8px;display:block}}
  .no-img{{background:#eee;height:80px;border-radius:4px;display:flex;
           align-items:center;justify-content:center;color:#999;margin-bottom:8px}}
  table{{width:100%;font-size:.82em;border-collapse:collapse}}
  td{{padding:3px 6px;vertical-align:top}}
  .lbl{{font-weight:700;color:#555;width:44px;white-space:nowrap}}
  tr:nth-child(even){{background:#f7f7f7}}
  .badge{{font-size:.78em;padding:2px 7px;border-radius:10px;white-space:nowrap;font-weight:700}}
  .pass-badge{{background:#d5f5e3;color:#1e8449}}
  .fail-badge{{background:#fadbd8;color:#c0392b}}
</style>
</head>
<body>
<h1>📋 日期辨識評估報告</h1>
<div class="stats">
  <div class="stat"><div class="val grey">{stats['total']}</div><div class="lbl2">圖片數（含日期）</div></div>
  <div class="stat"><div class="val blue">{stats['det_recall']:.1%}</div><div class="lbl2">Detection Recall</div></div>
  <div class="stat"><div class="val {'green' if stats['e2e_acc'] >= .8 else 'red'}">{stats['e2e_acc']:.1%}</div><div class="lbl2">E2E Accuracy</div></div>
  <div class="stat"><div class="val green">{pass_n}</div><div class="lbl2">正確</div></div>
  <div class="stat"><div class="val red">{fail_n}</div><div class="lbl2">錯誤</div></div>
  <div class="stat"><div class="val red">{nodet_n}</div><div class="lbl2">未偵測到</div></div>
</div>
<div class="filters">
  <span>顯示：</span>
  <button class="btn-all"  onclick="filter('all')">全部 ({len(records)})</button>
  <button class="btn-fail" onclick="filter('fail')">失敗 ({fail_n})</button>
  <button class="btn-pass" onclick="filter('pass')">正確 ({pass_n})</button>
</div>
<div class="grid" id="grid">
{''.join(cards)}
</div>
<script>
function filter(t){{
  document.querySelectorAll('.card').forEach(c=>{{
    c.style.display = (t==='all' || c.dataset.status===t) ? '' : 'none';
  }});
}}
</script>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report → {out_path}")


def load_models():
    print(f"Loading YOLO model: {YOLO_MODEL}")
    yolo = YOLO(str(YOLO_MODEL))
    if REC_MODEL and REC_MODEL.exists():
        print(f"Loading fine-tuned RapidOCR rec: {REC_MODEL}")
        keys_path = MODELS_DIR / "ppocr_keys_v1.txt"
        ocr = RapidOCR(rec_model_path=str(REC_MODEL),
                       rec_keys_path=str(keys_path) if keys_path.exists() else None)
    else:
        print("Loading RapidOCR default (PPOCRv4 mobile rec)...")
        ocr = RapidOCR()
    return yolo, ocr


def crop_bbox(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              padding: int = PADDING) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return img[y1:y2, x1:x2]


_MIN_REC_HEIGHT = 48   # PPOCRv4 is more stable at 48px than 32px


def _ensure_min_height(img: np.ndarray, target: int = _MIN_REC_HEIGHT) -> np.ndarray:
    h, w = img.shape[:2]
    if h < target:
        scale = target / h
        return cv2.resize(img, (max(1, int(w * scale)), target),
                          interpolation=cv2.INTER_CUBIC)
    return img


def _deskew(crop: np.ndarray) -> np.ndarray | None:
    """Estimate text skew angle and rotate to horizontal.
    Returns corrected image, or None if tilt < 2° (not worth rotating)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) < 50:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]   # in [-90, 0]
    if angle < -45:
        angle += 90   # normalize to [-45, 45]
    if abs(angle) < 2.0:
        return None
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(crop, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _preprocess_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Try variants in order; stop as soon as one yields a date."""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    block = max(11, (h // 4) * 2 + 1)
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 4
    )

    def to_bgr(g):
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    variants = [
        crop,                            # 1. original
        to_bgr(enhanced),                # 2. CLAHE contrast boost
        to_bgr(otsu),                    # 3. Otsu binary (dark-on-light)
        to_bgr(cv2.bitwise_not(otsu)),   # 4. inverted Otsu (light-on-dark)
        to_bgr(adapt),                   # 5. adaptive threshold
    ]

    # 6-7: deskewed variants (handles tilted labels)
    deskewed = _deskew(crop)
    if deskewed is not None:
        gray_d = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
        _, otsu_d = cv2.threshold(gray_d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(deskewed)             # 6. deskewed original
        variants.append(to_bgr(otsu_d))       # 7. deskewed + Otsu

    # 8-9: 2x upscale versions (helps for small/blurry crops)
    if h < 96:
        up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        gray_up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        _, otsu_up = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(up)                   # 8. 2x bilinear
        variants.append(to_bgr(otsu_up))      # 9. 2x + Otsu

    return variants


def recognize_crop(ocr: RapidOCR, crop: np.ndarray) -> str:
    if crop.size == 0:
        return ""
    crop = _ensure_min_height(crop)
    best_text = ""
    for variant in _preprocess_variants(crop):
        try:
            result, _ = ocr(variant, use_det=False, use_cls=False, use_rec=True)
            if result:
                text = " ".join(r[0] for r in result if r and isinstance(r[0], str))
                if extract_date(text):
                    return text        # early exit: date found
                if not best_text:
                    best_text = text   # keep first result as fallback
        except Exception:
            pass
    return best_text


# ── single-image inference ────────────────────────────────────────────────────

def run_image(img_path: str | Path, yolo: YOLO, ocr: RapidOCR,
              save: bool = False) -> list[dict]:
    """
    Returns list of dicts:
      { "bbox": [x1,y1,x2,y2], "conf": float, "rec": str, "date": str|None }
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return []

    det_results = yolo(img, conf=CONF_THRESH, verbose=False)[0]
    boxes = det_results.boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        crop = crop_bbox(img, x1, y1, x2, y2)
        rec_text = recognize_crop(ocr, crop)
        date_str = extract_date(rec_text)
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": round(conf, 3),
            "rec":  rec_text,
            "date": date_str,
        })

    if save:
        out = img.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["date"] or d["rec"] or "?"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        out_path = RESULTS_DIR / f"{img_path.stem}_pipeline.jpg"
        cv2.imwrite(str(out_path), out)
        print(f"  Saved → {out_path}")

    return detections


# ── batch eval ───────────────────────────────────────────────────────────────

def run_eval_val(yolo: YOLO, ocr: RapidOCR, report: bool = False,
                 split: str = "val"):
    label_path = DATASET_DIR / f"{split}_label.txt"
    img_dir    = DATASET_DIR / split

    total = tp_det = tp_e2e = fn_det = 0
    records: list[dict] = []

    with open(label_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc=f"eval {split}"):
        tab = line.index("\t")
        img_name = line[:tab]
        annotations = json.loads(line[tab + 1:])
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        gt_dates = [
            ann["transcription"]
            for ann in annotations
            if ann.get("transcription", "") != "###"
            and is_date_text(ann.get("transcription", ""))
        ]
        if not gt_dates:
            continue

        total += 1
        orig_img   = cv2.imread(str(img_path)) if report else None
        detections = run_image(img_path, yolo, ocr)

        if not detections:
            fn_det += 1
            matched = False
        else:
            tp_det += 1
            pred_dates = [d["date"] for d in detections if d["date"]]
            matched = False
            for pred in pred_dates:
                pred_d = _digits_only(pred)
                for gt in gt_dates:
                    gt_norm = extract_date(gt) or gt
                    gt_d    = _digits_only(gt_norm)
                    if pred_d and gt_d and pred_d == gt_d:
                        matched = True
                        break
                if matched:
                    break

        if matched:
            tp_e2e += 1

        if report:
            b64 = ""
            if orig_img is not None:
                b64 = _img_to_b64(_draw_report_boxes(orig_img, detections))
            records.append({
                "img":        img_name,
                "matched":    matched,
                "no_det":     not detections,
                "gt_dates":   gt_dates,
                "pred_dates": [d["date"] for d in detections if d["date"]],
                "rec_texts":  [d["rec"] for d in detections],
                "b64":        b64,
            })

    det_recall = tp_det / total if total else 0
    e2e_acc    = tp_e2e / total if total else 0

    print(f"\n[Val eval]  images_with_dates={total}")
    print(f"  Detection recall : {det_recall:.1%}  ({tp_det}/{total})")
    print(f"  E2E accuracy     : {e2e_acc:.1%}  ({tp_e2e}/{total})")
    print(f"  Missed entirely  : {fn_det}")

    if report:
        report_path = RESULTS_DIR / f"report_{split}.html"
        _generate_html_report(
            records,
            {"total": total, "det_recall": det_recall, "e2e_acc": e2e_acc,
             "tp_det": tp_det, "tp_e2e": tp_e2e, "fn_det": fn_det},
            report_path,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Date recognition pipeline")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--eval-val",   action="store_true", help="Eval on val set")
    parser.add_argument("--eval-train", action="store_true", help="Eval on train set")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML report alongside eval")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output image")
    args = parser.parse_args()

    yolo, ocr = load_models()

    if args.eval_val:
        run_eval_val(yolo, ocr, report=args.report, split="val")
    if args.eval_train:
        run_eval_val(yolo, ocr, report=args.report, split="train")
    elif args.image:
        dets = run_image(args.image, yolo, ocr, save=args.save)
        if not dets:
            print("No date regions detected.")
        else:
            print(f"\nDetected {len(dets)} date region(s):")
            for i, d in enumerate(dets, 1):
                print(f"  [{i}] conf={d['conf']:.2f}  rec='{d['rec']}'  "
                      f"date='{d['date']}'  bbox={d['bbox']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
