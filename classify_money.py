import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "money" / "weights" / "best.pt"
ALTERNATE_MODEL_PATH = (
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "money" / "weights" / "best.pt"
)
GENERAL_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25
INFERENCE_SIZE = 320

_money_model = None
_general_model = None


def _get_money_model():
    global _money_model
    if _money_model is not None:
        return _money_model

    model_path = Path(os.environ.get("MONEY_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists() and ALTERNATE_MODEL_PATH.exists():
        model_path = ALTERNATE_MODEL_PATH
    if not model_path.exists():
        raise RuntimeError(
            f"Currency model not found at {model_path}. Train it with "
            "yolo detect train model=yolov8n.pt "
            "data='my_dataset/Vietnamese Currency.v5-clean.yolov8/data.yaml' "
            "project=runs/detect name=money."
        )

    _money_model = YOLO(str(model_path))
    return _money_model


def _get_general_model():
    global _general_model
    if _general_model is None:
        _general_model = YOLO(str(GENERAL_MODEL_PATH))
    return _general_model


def _predictions(results):
    predictions = []
    for result in results:
        if result.boxes is None:
            continue
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            x_min, y_min, x_max, y_max = box
            predictions.append(
                {
                    "class": result.names[int(class_id)],
                    "confidence": float(confidence),
                    "x": (x_min + x_max) / 2,
                    "y": (y_min + y_max) / 2,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                }
            )
    return predictions


def classify_money(image):
    """Detect people, common objects, and currency in an OpenCV image."""
    general_results = _get_general_model().predict(
        source=image,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )
    money_results = _get_money_model().predict(
        source=image,
        imgsz=INFERENCE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )
    return {"predictions": _predictions(general_results) + _predictions(money_results)}
