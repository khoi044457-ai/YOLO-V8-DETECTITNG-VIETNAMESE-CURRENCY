import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "runs" / "classify" / "banknote" / "weights" / "best.pt"
TEST_DATASET = PROJECT_ROOT / "my_dataset" / "bank_note_cls" / "test"
OUTPUT_PATH = PROJECT_ROOT / "runs" / "classify" / "banknote" / "evaluation.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CONDITIONS = {
    "normal": lambda image: image,
    "dark": lambda image: cv2.convertScaleAbs(image, alpha=0.55, beta=-20),
    "bright": lambda image: cv2.convertScaleAbs(image, alpha=1.25, beta=20),
    "blurred": lambda image: cv2.GaussianBlur(image, ( nine := 9, nine), 0),
    "small": lambda image: cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA),
}


def image_paths():
    for class_directory in sorted(path for path in TEST_DATASET.iterdir() if path.is_dir()):
        for image_path in sorted(class_directory.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_directory.name, image_path


def evaluate_condition(model, condition_name, transform):
    confusion = Counter()
    total_latency = 0.0
    confidence_sum = 0.0
    total = 0

    for actual, image_path in image_paths():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        transformed = transform(image)
        start = time.perf_counter()
        result = model.predict(source=transformed, imgsz=224, verbose=False)[0]
        total_latency += time.perf_counter() - start
        predicted = model.names[int(result.probs.top1)]
        confidence_sum += float(result.probs.top1conf)
        confusion[(actual, predicted)] += 1
        total += 1

    classes = sorted({actual for actual, _ in confusion})
    per_class = {}
    for class_name in classes:
        true_positive = confusion[(class_name, class_name)]
        actual_count = sum(value for (actual, _), value in confusion.items() if actual == class_name)
        predicted_count = sum(value for (_, predicted), value in confusion.items() if predicted == class_name)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count if actual_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": true_positive / actual_count if actual_count else 0.0,
            "samples": actual_count,
        }

    correct = sum(value for (actual, predicted), value in confusion.items() if actual == predicted)
    return {
        "condition": condition_name,
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "average_confidence": confidence_sum / total if total else 0.0,
        "average_latency_ms": total_latency / total * 1000 if total else 0.0,
        "per_class": per_class,
        "confusion": {f"{actual}->{predicted}": value for (actual, predicted), value in sorted(confusion.items()) if actual != predicted},
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {MODEL_PATH}")
    if not TEST_DATASET.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DATASET}")

    model = YOLO(str(MODEL_PATH))
    report = {
        "model": str(MODEL_PATH),
        "dataset": str(TEST_DATASET),
        "conditions": [evaluate_condition(model, name, transform) for name, transform in CONDITIONS.items()],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for condition in report["conditions"]:
        print(
            f"{condition['condition']}: accuracy={condition['accuracy']:.2%}, "
            f"confidence={condition['average_confidence']:.2%}, "
            f"latency={condition['average_latency_ms']:.1f} ms"
        )
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
