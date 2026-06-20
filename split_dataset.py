import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================
RAW_DIR = Path("data/phone_detection/raw")
SPLIT_DIR = Path("data/phone_detection/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# SETUP
# =========================
random.seed(SEED)

# Remove old split folder if exists (to avoid duplicates)
if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

# Create new split folders
for split in ["train", "val", "test"]:
    (SPLIT_DIR / split).mkdir(parents=True, exist_ok=True)

# =========================
# SPLIT DATA
# =========================
class_names = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]

print("Classes found:", class_names)

for class_name in class_names:
    class_path = RAW_DIR / class_name

    images = [
        f for f in class_path.iterdir()
        if f.suffix.lower() in VALID_EXTENSIONS
    ]

    random.shuffle(images)

    total = len(images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count + val_count]
    test_imgs = images[train_count + val_count:]

    # Create class folders inside each split
    for split in ["train", "val", "test"]:
        (SPLIT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy2(img, SPLIT_DIR / "train" / class_name / img.name)

    for img in val_imgs:
        shutil.copy2(img, SPLIT_DIR / "val" / class_name / img.name)

    for img in test_imgs:
        shutil.copy2(img, SPLIT_DIR / "test" / class_name / img.name)

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_imgs)}")
    print(f"Val:   {len(val_imgs)}")
    print(f"Test:  {len(test_imgs)}")

print("\n✅ Dataset split completed successfully!")