# 訓練流程

## 資料集路徑

由 `yoloOcr/config.py` 自動推導：

| 資料集 | 預設路徑 | 說明 |
|--------|------|------|
| 原始偵測標注 | `<root>/dataset/det/` | PaddleOCR det 格式（train/val + label.txt） |
| YOLO 格式 | `<root>/yoloOcr/yolo_dataset/` | 由 `convert_to_yolo.py` 產生 |
| Rec 訓練資料 | `<root>/dataset/rec/` | PaddleOCR rec 格式 |

換機器只需設 `YOLOOCR_DATASET` 環境變數，不必改 script。

---

## Part A：YOLO 偵測模型（YOLO11n）

### 1) 本機：轉換成 YOLO 格式

```powershell
cd C:\Users\andy_ac_chen\Desktop\claudeProject
python yoloOcr\scripts\convert_to_yolo.py
# → yoloOcr/yolo_dataset/{images,labels}/{train,val}/ + data.yaml
```

### 2) 本機：打包上傳 Colab

```python
import zipfile
from pathlib import Path
SRC = Path("yoloOcr/yolo_dataset")
with zipfile.ZipFile("yolo_dataset.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in SRC.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(SRC.parent))
```

上傳 `yolo_dataset.zip` 到 `MyDrive/ocr_project/`。

### 3) Colab：訓練 YOLO11n（GPU）

開 `yoloOcr/colab_train_yolo.ipynb`，跑全部 cell。

主要參數：
- `model = YOLO("yolo11n.pt")`
- `epochs=100, imgsz=640, batch=32, device=0, patience=20`
- augmentation 沿用：`hsv_h/s/v`、`degrees=5`、`fliplr=0.5`、`mosaic=1.0`

### 4) 下載 `best.pt` → 覆蓋 `yoloOcr/models/best.pt`

預期 val mAP50 ≥ 0.95（baseline v8 為 0.972）。

---

## Part B：PPOCRv4 Rec 模型

**沿用現有 `yoloOcr/models/rec_finetuned.onnx`**，本次不重訓。

若未來想重訓，主線專案的 `train_rec_v5_colab.ipynb` 已涵蓋（見根目錄 `CLAUDE.md`）。

---

## Part C：本機推論 + Report

```powershell
python yoloOcr\scripts\pipeline.py --eval-val --report
# → yoloOcr/results/report.html
```

---

## Part D：Hard Negative Mining（迭代優化，可選）

```powershell
python yoloOcr\scripts\mine_failures.py
# → yoloOcr/rec_hard_negatives/{train,val}/ + label.txt
```

Merge 進 rec_dataset 時請用 Python（PowerShell `Add-Content` 會把 UTF-8 轉成 UTF-16 損壞標注）。

---

## 模型版本紀錄

| 版本 | 偵測模型 | rec 模型 | Val 圖數 | Val E2E |
|------|----------|---------|---------|---------|
| v1（原機） | YOLOv8n | rec_finetuned v2（舊機資料） | 352 | 84.4% |
| v2.0（本機初跑） | YOLO11n（2305 train） | rec_finetuned v2（舊機資料） | 577 | 76.4% |
| v2.1（換 rec） | YOLO11n | **rec_finetuned 91.86%**（本機資料訓） | 577 | **88.9%** |
| **v2.2（regex 加 YYYY-MMDD / 純年月）** | YOLO11n | rec_finetuned 91.86% | 577 | **90.8%** ✅ |

### v2.2 結果（2026-05-14）

- E2E **90.8%**（524/577）
- Detection Recall 100%（IoU≥0.5: 97%）
- 失敗 53 件 = wrong_length 29 / partial 9 / digit_1or2 8 / digit_more 4 / no_date 3

### 變更摘要（v2.0 → v2.2）

1. **rec ONNX 換新**：用 `output/rec/export_rec_onnx_double_click.bat` 把本機 `best_accuracy.pdparams`（91.86%）匯出，覆蓋 `models/rec_finetuned.onnx`
2. **`date_patterns.py` 加兩個 pattern**：`YYYY-MMDD` 緊湊式（`2025-1030`）+ `_YEAR_MONTH` 加入 EXTRACT
3. **PADDING 嘗試改 20 後回退**：12 仍是最佳值（大 padding 帶進雜訊反而傷 rec）

### v2.0 → v2.1 → v2.2 過程的舊備份

- `models/rec_finetuned.v2_oldmachine.onnx.bak`：v1/v2.0 用的舊機 rec（83.8% E2E）
