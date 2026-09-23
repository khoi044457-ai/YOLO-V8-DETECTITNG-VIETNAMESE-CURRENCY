import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATASET = PROJECT_ROOT / "my_dataset" / "Vietnamese Currency.v5-hope.yolov8 (1)"
OUTPUT_DATASET = PROJECT_ROOT / "my_dataset" / "Vietnamese Currency.v5-clean.yolov8"


def polygon_to_box(values):
    coordinates = values[1:]
    x_values = coordinates[0::2]
    y_values = coordinates[1::2]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return [values[0], (x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]


def convert_labels(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for label_path in source_dir.glob("*.txt"):
        converted = []
        for line in label_path.read_text().splitlines():
            values = [float(value) for value in line.split()]
            if len(values) == 5:
                box = values
            elif len(values) >= 7 and len(values[1:]) % 2 == 0:
                box = polygon_to_box(values)
            else:
                raise ValueError(f"Unsupported annotation in {label_path}: {line}")
            converted.append(" ".join(f"{value:.8f}" for value in box))
        (output_dir / label_path.name).write_text("\n".join(converted) + "\n")


def main():
    config = yaml.safe_load((SOURCE_DATASET / "data.yaml").read_text())
    for split in ("train", "valid", "test"):
        source_split = SOURCE_DATASET / split
        output_split = OUTPUT_DATASET / split
        if not source_split.exists():
            continue
        shutil.copytree(source_split / "images", output_split / "images", dirs_exist_ok=True)
        convert_labels(source_split / "labels", output_split / "labels")

    (OUTPUT_DATASET / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": config["nc"],
                "names": config["names"],
            },
            sort_keys=False,
        )
    )
    print(f"Clean detection dataset written to {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()
