# 換電腦設定指南

## 1. Clone 專案

```bash
git clone https://github.com/install88/trainingOcr.git
cd trainingOcr/yoloOcr
```

## 2. 安裝 Python 環境

Python 3.10+ 建議。**強烈建議使用 venv 隔離**，因為 ultralytics 會拉 torch（~2GB）且版本綁很死。

```powershell
cd yoloOcr
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

之後每次跑 yoloOcr script 前先 `venv\Scripts\activate`（主線 PPOCRv4 流程不受影響）。

## 3. 資料集路徑

預設讀取 `<專案根>/dataset/det/`（PaddleOCR det 格式：`train/`, `val/`, `train_label.txt`, `val_label.txt`）。

**所有路徑由 `yoloOcr/config.py` 統一管理，不需要改 script。** 若資料集不在預設位置，設環境變數：

```powershell
$env:YOLOOCR_DATASET = "D:\path\to\det_dataset"
```

可用環境變數：

| 變數 | 預設 | 用途 |
|------|------|------|
| `YOLOOCR_DATASET` | `<root>/dataset/det` | PaddleOCR det 格式資料集 |
| `YOLOOCR_MODELS` | `yoloOcr/models` | 模型檔位置 |
| `YOLOOCR_RESULTS` | `yoloOcr/results` | report.html 輸出 |
| `YOLOOCR_YOLO_DS` | `yoloOcr/yolo_dataset` | 轉換後 YOLO 格式資料集 |
| `YOLOOCR_RUNS` | `yoloOcr/runs` | 訓練輸出 |

## 4. 確認模型檔案

`yoloOcr/models/` 應包含：

```
models/
├── best.pt              # YOLO 偵測模型（baseline；新訓後覆蓋）
├── rec_finetuned.onnx   # PPOCRv4 rec ONNX
└── ppocr_keys_v1.txt    # 字典
```

## 5. 測試環境

```bash
python yoloOcr/scripts/pipeline.py --eval-val
```

## 6. 重新訓練 YOLO 偵測模型

見 [TRAINING.md](TRAINING.md)。建議用 Colab GPU。
