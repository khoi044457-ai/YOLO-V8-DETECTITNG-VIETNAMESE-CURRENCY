import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = PROJECT_ROOT / "my_dataset" / "Vietnamese Currency.v5-clean.yolov8" / "data.yaml"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Currency dataset config not found: {DATASET}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATASET),
        epochs=int(os.environ.get("MONEY_EPOCHS", "20")),
        imgsz=int(os.environ.get("MONEY_IMAGE_SIZE", "320")),
        batch=int(os.environ.get("MONEY_BATCH", "1")),
        device="cpu",
        workers=0,
        amp=False,
        plots=False,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="money_vietnamese",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()