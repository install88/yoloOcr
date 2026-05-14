"""
Step 4: Train YOLOv8n on the converted date-region dataset.
Run after convert_to_yolo.py completes.
"""
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO
from config import YOLO_DATASET_DIR as YOLO_DATASET, RUNS_DIR as OUTPUT_DIR


def main():
    data_yaml = YOLO_DATASET / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Run convert_to_yolo.py first. Missing: {data_yaml}")

    model = YOLO("yolo11n.pt")  # YOLO11n (Ultralytics 2024-09); ~2.6M params

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=32,
        patience=20,          # early stopping
        project=str(OUTPUT_DIR),
        name="date_det_v2",
        exist_ok=True,
        device=0,             # GPU; set to "cpu" if no CUDA
        workers=4,
        cache=False,
        augment=True,
        # augmentation params (helps with varied packaging)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,          # slight rotation (labels can be tilted)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best model: {best}")

    # Quick validation
    val_results = model.val(data=str(data_yaml))
    print(f"Val mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Val mAP@0.5:0.95: {val_results.box.map:.4f}")

    # Export to NCNN for Android
    print("\nExporting to NCNN...")
    model.export(format="ncnn", imgsz=640)
    print("NCNN export done.")


if __name__ == "__main__":
    main()
