# 商品包裝日期辨識系統

手機拍照 → 自動辨識商品包裝上的到期日 / 製造日。

## 架構

```
相機幀
  ↓
[YOLOv8n]  偵測「日期文字區域」bounding box
  ↓
裁切 + 多版本前處理（CLAHE / Otsu / Deskew）
  ↓
[PPOCRv4_mobile_rec (ONNX)]  辨識文字
  ↓
[Regex 解析]  提取標準化日期字串
  ↓
輸出結果
```

## 目前成效（val set，352 張）

| 指標 | 數值 |
|------|------|
| YOLO Detection Recall | 99.7% |
| E2E Accuracy | **84.4%** |
| YOLO mAP50 | 0.972 |

## 專案結構

```
date-recognition/
├── scripts/
│   ├── pipeline.py          # 主要推論流程（YOLO → 前處理 → OCR → Regex）
│   ├── date_patterns.py     # 日期 Regex patterns（is_date_text / extract_date）
│   ├── baseline_rec.py      # Step 2: 只跑 rec 的 baseline 評估
│   ├── convert_to_yolo.py   # Step 3: PaddleOCR det 標注 → YOLO 格式
│   ├── train_yolo.py        # Step 4: YOLOv8 訓練
│   ├── mine_failures.py     # 從失敗 case 挖出 hard negative crops
│   └── analyze_dataset.py   # 資料集統計分析
├── models/
│   ├── best.pt              # YOLO 偵測模型（6.2MB）
│   ├── rec_finetuned.onnx   # PPOCRv4 rec 模型（fine-tuned, 10.4MB）
│   └── ppocr_keys_v1.txt    # 字典（6623 字）
├── colab_train.ipynb        # Colab：訓練 PPOCRv4 rec
├── colab_export_rec.ipynb   # Colab：pdparams → ONNX 轉換
├── requirements.txt
└── README.md
```

## 快速開始

```bash
pip install -r requirements.txt

# 單張圖片推論
python scripts/pipeline.py path/to/image.jpg --save

# 評估 val set
python scripts/pipeline.py --eval-val

# 產生 HTML 報告
python scripts/pipeline.py --eval-val --report
```

## 訓練流程

完整訓練步驟見 [TRAINING.md](TRAINING.md)
