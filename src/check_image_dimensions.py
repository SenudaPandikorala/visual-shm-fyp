
from pathlib import Path
from collections import Counter

import cv2
import numpy as np


DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/"
    "Reinforced concrete structure segmentation dataset"
)

IMAGE_DIR = DATASET_PATH / "imagedata"
MASK_DIR = DATASET_PATH / "maskdata"

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

image_sizes = Counter()
mask_sizes = Counter()
problems = []
mask_values = set()

print("Checking images and masks...\n")

for image_path in image_files:
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        problems.append(f"Cannot read image: {image_path.name}")
        continue

    if mask is None:
        problems.append(f"Cannot read mask: {mask_path.name}")
        continue

    # OpenCV shapes are (height, width, channels)
    image_height, image_width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    image_sizes[(image_width, image_height)] += 1
    mask_sizes[(mask_width, mask_height)] += 1

    if (image_height, image_width) != (mask_height, mask_width):
        problems.append(
            f"Dimension mismatch: {image_path.name} "
            f"image={image_width}x{image_height}, "
            f"mask={mask_width}x{mask_height}"
        )

    if mask.ndim != 2:
        problems.append(
            f"Mask is not single-channel: {mask_path.name}, "
            f"shape={mask.shape}"
        )
        continue

    mask_values.update(np.unique(mask).tolist())


print("===== DATASET DIMENSIONS =====")
print("Readable images:", sum(image_sizes.values()))
print("Distinct image dimensions:", len(image_sizes))
print("Distinct mask dimensions:", len(mask_sizes))

print("\n===== MOST COMMON IMAGE SIZES =====")
for size, count in image_sizes.most_common(10):
    print(f"{size[0]} x {size[1]}: {count} images")

print("\n===== MASK CLASS VALUES =====")
print("Values found:", sorted(mask_values))

print("\n===== VALIDATION ISSUES =====")
if problems:
    for problem in problems:
        print(problem)
    print("Total issues:", len(problems))
else:
    print("No image/mask dimension or file-format issues found.")