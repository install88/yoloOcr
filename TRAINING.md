# 訓練流程

## 資料集路徑（本機）

| 資料集 | 路徑 | 說明 |
|--------|------|------|
| 原始偵測標注 | `C:\Users\insta\Desktop\dataset\` | PaddleOCR det 格式，含 train/val label |
| YOLO 格式 | `C:\Users\insta\Desktop\date-recognition\yolo_dataset\` | 由 convert_to_yolo.py 產生 |
| Rec 訓練資料 | `C:\Users\insta\Desktop\dataset\rec_dataset\` | PaddleOCR rec 格式（2046 train / 519 val）|

---

## Part A：YOLO 偵測模型

### 準備資料
```bash
python scripts/convert_to_yolo.py
# 輸出：yolo_dataset/images/ + labels/ + data.yaml
```

### 訓練
```bash
python scripts/train_yolo.py
# 或直接用 ultralytics CLI：
yolo detect train model=yolov8n.pt data=yolo_dataset/data.yaml epochs=100 imgsz=640 batch=16
```

### 結果
- 模型輸出：`runs/date_det_v1/weights/best.pt`
- mAP50 = 0.972，mAP50-95 = 0.790

---

## Part B：PPOCRv4 Rec 模型（Colab 訓練）

### 訓練 notebook
[train_rec_colab.ipynb](https://colab.research.google.com/github/install88/trainingOcr/blob/main/notebooks/train_rec_colab.ipynb)

### 訓練資料
使用 `C:\Users\insta\Desktop\dataset\rec_dataset\`（已包含 hard negative crops）

Hard negative mining 後用 Python 重新打包（見 Part C），上傳到 Google Drive：`MyDrive/ocr_project/rec_dataset_v2.zip`

### 訓練完成後輸出
- `model.pdparams`（PaddlePaddle 格式，約 10MB）

### 轉換 ONNX
使用 [colab_export_rec.ipynb](colab_export_rec.ipynb)：
1. 上傳 `model.pdparams`
2. 跑全部 cell
3. 下載 `rec_finetuned.onnx`（約 10.4MB）
4. 複製到 `models/rec_finetuned.onnx`

---

## Part C：Hard Negative Mining（迭代優化）

每次訓練後，從當前失敗 case 挖出新訓練樣本：

```bash
python scripts/mine_failures.py
# 輸出：rec_hard_negatives/train/ + val/ + 對應 label.txt
```

merge 到 rec_dataset（**必須用 Python**，PowerShell `Add-Content` 會把 UTF-8 轉成 UTF-16 導致標注檔損壞）：

```python
# merge_hard_negatives.py
from pathlib import Path

REC_DIR = Path(r"C:\Users\insta\Desktop\dataset\rec_dataset")
HN_DIR  = Path(r"C:\Users\insta\Desktop\date-recognition\rec_hard_negatives")

for split in ["train", "val"]:
    orig = (REC_DIR / f"{split}_label.txt").read_text(encoding="utf-8").splitlines()
    hn   = (HN_DIR  / f"{split}_label.txt").read_text(encoding="utf-8").splitlines()
    merged = [l for l in orig + hn if l.strip()]
    (REC_DIR / f"{split}_label.txt").write_text("\n".join(merged), encoding="utf-8")
    print(f"{split}: {len(merged)} lines")

import shutil
for img in (HN_DIR / "train").iterdir():
    shutil.copy2(img, REC_DIR / "train" / img.name)
for img in (HN_DIR / "val").iterdir():
    shutil.copy2(img, REC_DIR / "val" / img.name)
```

merge 後重新打包：
```python
import zipfile
from pathlib import Path
REC_DIR = Path(r"C:\Users\insta\Desktop\dataset\rec_dataset")
with zipfile.ZipFile(r"C:\Users\insta\Desktop\dataset\rec_dataset_v2.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["train_label.txt", "val_label.txt"]:
        z.write(REC_DIR / f, f)
    for split in ["train", "val"]:
        for img in (REC_DIR / split).iterdir():
            z.write(img, f"{split}/{img.name}")
```

---

## 模型版本紀錄

| 版本 | rec 模型 | Val E2E |
|------|----------|---------|
| v1 | RapidOCR 預設（未 fine-tune） | 56.9% |
| v2 | 第一版 fine-tune（rec_dataset 1846 train） | 83.8% |
| v2.1 | + Deskew 前處理 + E-class regex 修正 | 84.4% |
| v3 | rec_dataset v2（2046 train / 519 val，含 hard negatives），從 v2 fine-tune 20 epochs | 訓練中 |
