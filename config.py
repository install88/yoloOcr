"""Path configuration for yoloOcr pipeline.

All paths are derived from this file's location so the project can be cloned
to any directory without code changes. Override via environment variables if
your dataset/models live elsewhere.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR     = Path(__file__).resolve().parent

DATASET_DIR      = Path(os.getenv("YOLOOCR_DATASET",  PROJECT_ROOT / "dataset" / "det"))
MODELS_DIR       = Path(os.getenv("YOLOOCR_MODELS",   YOLO_DIR / "models"))
RESULTS_DIR      = Path(os.getenv("YOLOOCR_RESULTS",  YOLO_DIR / "results"))
YOLO_DATASET_DIR = Path(os.getenv("YOLOOCR_YOLO_DS",  YOLO_DIR / "yolo_dataset"))
RUNS_DIR         = Path(os.getenv("YOLOOCR_RUNS",     YOLO_DIR / "runs"))

YOLO_MODEL = MODELS_DIR / "best.pt"
REC_MODEL  = MODELS_DIR / "rec_finetuned.onnx"
REC_KEYS   = MODELS_DIR / "ppocr_keys_v1.txt"
