import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = PROJECT_ROOT / "my_dataset" / "bank_note_cls"
MODEL_PATH = PROJECT_ROOT / "yolov8n-cls.pt"
RUN_DIRECTORY = PROJECT_ROOT / "runs" / "classify" / "banknote"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Classification dataset not found: {DATASET}. "
            "Run prepare_banknote_classification.py first."
        )

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    checkpoint = RUN_DIRECTORY / "weights" / "last.pt"
    resume = checkpoint.exists()
    model = YOLO(str(checkpoint if resume else MODEL_PATH))
    model.train(
        data=str(DATASET),
        task="classify",
        epochs=int(os.environ.get("BANKNOTE_EPOCHS", "20")),
        imgsz=int(os.environ.get("BANKNOTE_IMAGE_SIZE", "224")),
        batch=int(os.environ.get("BANKNOTE_BATCH", "8")),
        device=os.environ.get("BANKNOTE_DEVICE", "cpu"),
        workers=0,
        amp=False,
        plots=False,
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name="banknote",
        exist_ok=True,
        resume=resume,
    )


if __name__ == "__main__":
    main()