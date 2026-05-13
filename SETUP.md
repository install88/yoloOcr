# 換電腦設定指南

## 1. Clone 專案

```bash
git clone https://github.com/install88/yoloOcr.git
cd yoloOcr
```

## 2. 安裝 Python 環境

Python 3.10+ 建議。

```bash
pip install -r requirements.txt
```

## 3. 準備資料集

資料集**不在 git repo 裡**（太大），需要另外取得：

- `rec_dataset_v2.zip`：從 Google Drive 下載，解壓到 `C:\Users\<you>\Desktop\dataset\rec_dataset\`
- 原始偵測資料集：從 Google Drive 下載，解壓到 `C:\Users\<you>\Desktop\dataset\`

> 如果路徑不同，修改各 script 頂部的 `DATASET_DIR` 常數即可。

## 4. 確認模型檔案

`models/` 資料夾應包含：

```
models/
├── best.pt              # YOLO model（已在 repo）
├── rec_finetuned.onnx   # Rec model（已在 repo）
└── ppocr_keys_v1.txt    # 字典（已在 repo）
```

## 5. 測試環境

```bash
python scripts/pipeline.py --eval-val
```

正常應看到：`E2E accuracy : 84.4%`

## 6. 繼續訓練 rec 模型

參考 [TRAINING.md](TRAINING.md) Part B。
