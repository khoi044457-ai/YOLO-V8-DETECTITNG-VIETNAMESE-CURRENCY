import queue
import re
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import Foundation
import pyttsx3
import Vision
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MONEY_DATASET = PROJECT_ROOT / "my_dataset" / "Vietnamese Currency.v5-clean.yolov8"

MODEL_CANDIDATES = [
    PROJECT_ROOT / "runs" / "detect" / "money_vietnamese" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "money_probe" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "money_vietnamese" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "detect" / "money" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "detect" / "runs" / "coco128_cpu" / "weights" / "best.pt",
]
CLASSIFIER_PATH = PROJECT_ROOT / "runs" / "classify" / "banknote" / "weights" / "best.pt"


def is_valid_model(candidate):
    try:
        model = YOLO(str(candidate))
        dataset_dir = MONEY_DATASET / "valid" / "images"
        sample_images = [
            str(path)
            for path in sorted(dataset_dir.iterdir())
            if path.is_file()
        ]
        if not sample_images:
            return True

        results = model(sample_images[0], imgsz=320, conf=0.25, verbose=False)
        return any(len(result.boxes) > 0 for result in results)
    except Exception as error:
        print(f"Model validation failed for {candidate}: {error}")
        return False


def resolve_model_path():
    for candidate in MODEL_CANDIDATES:
        if candidate.exists() and is_valid_model(candidate):
            return str(candidate)
    general_model = PROJECT_ROOT / "yolov8n.pt"
    if general_model.exists():
        return str(general_model)
    return str(MODEL_CANDIDATES[-1])


MODEL_PATH = resolve_model_path()

CONFIDENCE_THRESHOLD = 0.15
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.50
TWENTY_THOUSAND_LABEL = "20000"
# Labels the classifier tends to give a 20k note by mistake; we double-check these.
TWENTY_THOUSAND_CONFUSABLE = {"2000", "5000", "50000", "200000", "500000"}
TWENTY_THOUSAND_MIN_SCORE = 0.65
DEBUG_DECISIONS = True  # prints detector/classifier/OCR results whenever the answer changes
SUPPORTED_VALUES = {"1000", "2000", "5000", "10000", "20000", "50000", "100000", "200000", "500000"}
INFERENCE_SIZE = 640

# Internal labels (dataset / classifier class names) -> short names shown and spoken.
DISPLAY_NAMES = {
    "1000": "1k",
    "2000": "2k",
    "5000": "5k",
    "10000": "10k",
    "20000": "20k",
    "50000": "50k",
    "100000": "100k",
    "200000": "200k",
    "500000": "500k",
}


def display_name(label):
    """Return the short name (e.g. '10k') for a label, or the label itself if unknown."""
    return DISPLAY_NAMES.get(str(label), str(label))

ANNOUNCEMENT_COOLDOWN = 3.0

# --- Number-reading (OCR) settings -------------------------------------------
OCR_HISTORY = 6          # how many recent OCR readings to remember
OCR_VOTES_NEEDED = 2     # same value must appear this many times to be trusted
OCR_MAX_AGE = 5.0        # seconds; older readings are ignored
OCR_MIN_WIDTH = 640      # small crops are upscaled to this width before OCR
OCR_ROTATIONS = (None, cv2.ROTATE_180)  # add cv2.ROTATE_90_CLOCKWISE etc. if needed

OCR_DEBUG = True  # prints what Vision reads so you can see why a note is missed

# OCR often confuses letters with digits on stylised banknote fonts (e.g. "1OOOO").
OCR_FIXES = str.maketrans("OoDQIl|", "0000111")

# Matches "10 000", "10.000", "10,000" or "10000" as a whole number, not part of a serial number.
NUMBER_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}(?:[.,\s]\d{3})+|\d{4,6})(?!\d)")


def _recognize_text(image_bgr):
    """Run macOS Vision text recognition on a BGR image and return the text lines."""
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        return []

    data = Foundation.NSData.dataWithBytes_length_(encoded.tobytes(), len(encoded))
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # we want raw digits, not "corrected" words
    request.setRecognitionLanguages_(["en-US"])

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    success, error = handler.performRequests_error_([request], None)
    if not success:
        return []

    lines = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if candidates:
            lines.append(str(candidates[0].string()))
    return lines


def _find_value(text):
    """Find a supported denomination inside one piece of OCR text."""
    text = text.translate(OCR_FIXES)
    for token in NUMBER_PATTERN.findall(text):
        value = re.sub(r"\D", "", token)
        if value in SUPPORTED_VALUES:
            return value
    return None


def _ocr_passes(image):
    """Yield (image, description) variants: original, contrast-enhanced, rotated."""
    yield image, "original"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), "contrast"
    for rotation in OCR_ROTATIONS:
        if rotation is not None:
            yield cv2.rotate(image, rotation), "rotated"


def read_money_value(image):
    """Read a supported denomination (e.g. '10000') printed on a BGR crop, or None."""
    if image is None or image.size == 0:
        return None

    width = image.shape[1]
    if width < OCR_MIN_WIDTH:
        scale = OCR_MIN_WIDTH / width
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    for variant, name in _ocr_passes(image):
        lines = _recognize_text(variant)
        if OCR_DEBUG and lines:
            print(f"OCR [{name}]: {lines}")
        # Try each line, then all lines joined (Vision may split "10" and "000").
        for text in [*lines, " ".join(lines)]:
            value = _find_value(text)
            if value:
                return value
    return None


class OcrWorker:
    """Reads numbers in a background thread and votes across recent frames."""

    def __init__(self):
        self.jobs = queue.Queue(maxsize=1)
        self.readings = deque(maxlen=OCR_HISTORY)  # (timestamp, value or None)
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            crop, full_frame = job
            try:
                # Try the note's box first, then the whole frame as a fallback.
                value = read_money_value(crop) or read_money_value(full_frame)
            except Exception as error:
                print(f"OCR error: {error}")
                value = None
            with self.lock:
                self.readings.append((time.monotonic(), value))

    def submit(self, crop, full_frame):
        """Queue a crop only if the worker is idle, so the camera never waits."""
        if crop is None or crop.size == 0 or not self.jobs.empty():
            return
        try:
            self.jobs.put_nowait((crop.copy(), full_frame.copy()))
        except queue.Full:
            pass

    def stable_value(self):
        """Return a denomination only if it was read consistently and recently."""
        cutoff = time.monotonic() - OCR_MAX_AGE
        with self.lock:
            values = [value for stamp, value in self.readings if stamp >= cutoff and value]
        if not values:
            return None
        value, count = Counter(values).most_common(1)[0]
        return value if count >= OCR_VOTES_NEEDED else None

    def stop(self):
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass


class SpeechWorker:
    """Speak the latest detection without blocking camera inference."""

    def __init__(self):
        self.messages = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
        except Exception as error:
            print(f"Audio disabled: {error}")
            return

        while True:
            message = self.messages.get()
            if message is None:
                return
            engine.say(message)
            engine.runAndWait()

    def speak(self, message):
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            pass

    def stop(self):
        try:
            self.messages.put_nowait(None)
        except queue.Full:
            pass


def classify_crop(classifier, crop):
    """Return (label, confidence) from the classifier, or None if unsure."""
    if classifier is None or crop.size == 0:
        return None
    probs = classifier.predict(source=crop, imgsz=224, device="cpu", verbose=False)[0].probs
    if probs is None:
        return None
    confidence = float(probs.top1conf)
    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
        return None
    return classifier.names[int(probs.top1)], confidence


def center_region(frame):
    height, width = frame.shape[:2]
    margin_x = int(width * 0.15)
    margin_y = int(height * 0.15)
    return margin_x, margin_y, width - margin_x, height - margin_y


def twenty_thousand_score(classifier, frame, twenty_index):
    """Highest 20k probability from the center crop and the full frame (0 if unavailable)."""
    if classifier is None or twenty_index is None:
        return 0.0
    x1, y1, x2, y2 = center_region(frame)
    scores = []
    for view in (frame[y1:y2, x1:x2], frame):
        probs = classifier.predict(source=view, imgsz=224, device="cpu", verbose=False)[0].probs
        if probs is not None:
            scores.append(float(probs.data[twenty_index]))
    return max(scores, default=0.0)


def main():
    detector = YOLO(MODEL_PATH)
    classifier = YOLO(str(CLASSIFIER_PATH)) if CLASSIFIER_PATH.exists() else None
    speech = SpeechWorker()
    ocr = OcrWorker()
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        speech.stop()
        ocr.stop()
        raise RuntimeError(
            "Cannot open camera. Grant camera access to VS Code or Terminal in "
            "System Settings > Privacy & Security > Camera."
        )

    # Index of the "20000" class in the classifier (looked up by name, not hard-coded).
    twenty_index = None
    if classifier is not None:
        twenty_index = next(
            (idx for idx, name in classifier.names.items() if name == TWENTY_THOUSAND_LABEL),
            None,
        )

    classifier_message = f" and classifier: {CLASSIFIER_PATH}" if classifier else ""
    print(f"Camera started with detector: {MODEL_PATH}{classifier_message}. Press q in the Smart Vision window to quit.")

    last_announced = {}
    last_decision = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read a frame from the camera. Check camera permissions and connection.")
                break

            results = detector.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=0.50,
                iou=0.35,
                max_det=1,
                agnostic_nms=True,
                device="cpu",
                verbose=False,
            )

            now = time.monotonic()
            frame_height, frame_width = frame.shape[:2]

            # ---- Way 1 + 2: YOLO detector, refined by the image classifier ----
            region = None   # (x1, y1, x2, y2) of the banknote
            label = None
            label_confidence = 0.0
            color = (0, 255, 0)

            for result in results:
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    region = (x1, y1, x2, y2)
                    label = detector.names[int(box.cls[0])]
                    label_confidence = confidence

                    classified = classify_crop(classifier, frame[y1:y2, x1:x2])
                    if classified is not None:
                        label, label_confidence = classified
                        # 20k is often confused with similar notes: double-check it.
                        if label in TWENTY_THOUSAND_CONFUSABLE:
                            score = twenty_thousand_score(classifier, frame, twenty_index)
                            if score >= TWENTY_THOUSAND_MIN_SCORE:
                                label = TWENTY_THOUSAND_LABEL
                                label_confidence = score
                    break
                if region is not None:
                    break

            # No box found: fall back to classifying the center of the frame.
            if region is None and classifier is not None:
                fx1, fy1, fx2, fy2 = center_region(frame)
                classified = classify_crop(classifier, frame[fy1:fy2, fx1:fx2])
                if classified is not None:
                    region = (fx1, fy1, fx2, fy2)
                    label, label_confidence = classified
                    color = (0, 255, 255)
                    if label in TWENTY_THOUSAND_CONFUSABLE:
                        score = twenty_thousand_score(classifier, frame, twenty_index)
                        if score >= TWENTY_THOUSAND_MIN_SCORE:
                            label = TWENTY_THOUSAND_LABEL
                            label_confidence = score

            # ---- Way 3: read the number printed on the note (OCR) ----
            ocr_region = region or center_region(frame)
            ox1, oy1, ox2, oy2 = ocr_region
            pad_x = int((ox2 - ox1) * 0.1)
            pad_y = int((oy2 - oy1) * 0.1)
            ocr.submit(
                frame[max(0, oy1 - pad_y):oy2 + pad_y, max(0, ox1 - pad_x):ox2 + pad_x],
                frame,
            )
            ocr_value = ocr.stable_value()
            model_label, model_confidence = label, label_confidence
            if ocr_value is not None:
                # OCR often drops a zero (reads "20000" as "2000"). If the model's label is the
                # same digits with more zeros, trust the model instead of the shorter OCR value.
                dropped_zero = (
                    model_label is not None
                    and model_label != ocr_value
                    and model_label.startswith(ocr_value)
                )
                if dropped_zero:
                    ocr_value = None
                else:
                    label = ocr_value
                    label_confidence = 1.0
                    if region is None:
                        region = ocr_region
                        color = (0, 255, 255)

            if DEBUG_DECISIONS and (model_label, ocr_value, label) != last_decision:
                last_decision = (model_label, ocr_value, label)
                print(
                    f"[decision] model={model_label} ({model_confidence:.0%}) "
                    f"ocr={ocr_value} -> final={label}"
                )

            # ---- Draw and announce ----
            if region is not None and label is not None:
                x1, y1, x2, y2 = region
                text = f"{display_name(label)} {label_confidence:.0%}"
                if ocr_value is not None:
                    text += " (OCR)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
                if now - last_announced.get(label, 0) >= ANNOUNCEMENT_COOLDOWN:
                    speech.speak(f"{display_name(label)} detected")
                    last_announced[label] = now

            cv2.imshow("Smart Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()
        ocr.stop()


if __name__ == "__main__":
    main()