import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DATASET = PROJECT_ROOT / "my_dataset" / "bank_note"
OUTPUT_DATASET = PROJECT_ROOT / "my_dataset" / "bank_note_cls"
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.8, 0.1, 0.1)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42


def main():
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Banknote dataset not found: {SOURCE_DATASET}")

    class_directories = sorted(path for path in SOURCE_DATASET.iterdir() if path.is_dir())
    if not class_directories:
        raise RuntimeError(f"No denomination folders found in {SOURCE_DATASET}")

    random_generator = random.Random(RANDOM_SEED)
    copied_images = 0
    for class_directory in class_directories:
        images = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        random_generator.shuffle(images)
        train_end = int(len(images) * SPLIT_RATIOS[0])
        val_end = train_end + int(len(images) * SPLIT_RATIOS[1])
        split_images = (
            ("train", images[:train_end]),
            ("val", images[train_end:val_end]),
            ("test", images[val_end:]),
        )

        for split, split_paths in split_images:
            destination = OUTPUT_DATASET / split / class_directory.name
            destination.mkdir(parents=True, exist_ok=True)
            for image_path in split_paths:
                shutil.copy2(image_path, destination / image_path.name)
                copied_images += 1

    print(
        f"Classification dataset written to {OUTPUT_DATASET} "
        f"({len(class_directories)} classes, {copied_images} images)"
    )


if __name__ == "__main__":
    main()