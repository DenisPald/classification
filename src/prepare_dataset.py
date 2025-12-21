import shutil
import random
from pathlib import Path
from collections import defaultdict


def prepare_dataset(
    raw_dir: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_images_per_class: int,
    seed: int,
):
    random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(exist_ok=True)

    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        images = list(class_dir.glob("*"))

        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, files in splits.items():
            target_dir = out_dir / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img in files:
                shutil.copy2(img, target_dir / img.name)


if __name__ == "__main__":
    prepare_dataset(
        raw_dir=Path("data/images/raw"),
        out_dir=Path("data/images/processed"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        min_images_per_class=20,
        seed=42,
    )
