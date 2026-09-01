import os
import random
import shutil

# =========================================
# CONFIG
# =========================================

SOURCE_DIR = r"C:\Users\guilh\instrument-vision-system\images"

OUTPUT_DIR = r"C:\Users\guilh\instrument-vision-system\dataset"

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

# =========================================
# CREATE FOLDERS
# =========================================

classes = os.listdir(SOURCE_DIR)

for split in SPLIT_RATIOS.keys():

    for cls in classes:

        path = os.path.join(OUTPUT_DIR, split, cls)

        os.makedirs(path, exist_ok=True)

# =========================================
# SPLIT FILES
# =========================================

for cls in classes:

    class_dir = os.path.join(SOURCE_DIR, cls)

    images = os.listdir(class_dir)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_images in split_data.items():

        for img in split_images:

            src = os.path.join(class_dir, img)

            dst = os.path.join(
                OUTPUT_DIR,
                split,
                cls,
                img
            )

            shutil.copy2(src, dst)

print("Dataset dividido com sucesso!")