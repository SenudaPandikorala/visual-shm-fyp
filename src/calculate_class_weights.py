
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

from dataset import IMAGE_DIR, MASK_DIR, SPLIT_DIR


NUM_CLASSES = 6
CLASS_NAMES = [
    "Background",
    "Crack",
    "Concrete spalling",
    "Rebar exposure",
    "Rebar corrosion",
    "Concrete crushing",
]

train_df = pd.read_csv(SPLIT_DIR / "train.csv")

pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

print("Counting pixels in training masks...")

for _, row in train_df.iterrows():

    mask_path = MASK_DIR / row["mask_filename"]

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED
    )

    if mask is None:
        raise FileNotFoundError(mask_path)

    values, counts = np.unique(
        mask,
        return_counts=True
    )

    for value, count in zip(values, counts):
        if value < 0 or value >= NUM_CLASSES:
            raise ValueError(
                f"Unexpected class ID {value} in {mask_path}"
            )

        pixel_counts[value] += count


# Inverse-square-root frequency weighting.
# This is a conservative starting point compared with
# using the full inverse frequency.
total_pixels = pixel_counts.sum()
frequencies = pixel_counts / total_pixels

weights = np.zeros(NUM_CLASSES, dtype=np.float64)

for class_id, frequency in enumerate(frequencies):
    if frequency > 0:
        weights[class_id] = 1.0 / np.sqrt(frequency)

# Normalize so the average class weight is 1.
weights = weights / weights.mean()

print("\n===== TRAINING PIXEL COUNTS =====")

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:22s} | "
        f"Pixels: {pixel_counts[i]:,} | "
        f"Frequency: {frequencies[i]:.6f} | "
        f"Weight: {weights[i]:.4f}"
    )

output_path = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "class_weights.npy"
)

output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, weights)

print("\nSaved class weights to:", output_path)
