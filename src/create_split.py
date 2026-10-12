
import csv
import json
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# --------------------------------------------------
# 1. Define dataset paths
# --------------------------------------------------

DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/"
    "Reinforced concrete structure segmentation dataset"
)

IMAGE_DIR = DATASET_PATH / "imagedata"
MASK_DIR = DATASET_PATH / "maskdata"
JSON_DIR = DATASET_PATH / "jsondata"

# Save split files inside our project.
PROJECT_PATH = Path(__file__).resolve().parent.parent
SPLIT_DIR = PROJECT_PATH / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate image-mask-JSON pairs
# --------------------------------------------------

samples = []

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_files:
    raise RuntimeError(f"No JPG images found in {IMAGE_DIR}")

for image_path in image_files:
    stem = image_path.stem

    mask_path = MASK_DIR / f"{stem}.png"
    json_path = JSON_DIR / f"{stem}.json"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    # Each image may contain one or more defect classes.
    labels = sorted({
        shape["label"]
        for shape in annotation_data["shapes"]
    })

    if not labels:
        raise ValueError(f"No defect labels found in {json_path}")

    samples.append({
        "filename": image_path.name,
        "mask_filename": mask_path.name,
        "json_filename": json_path.name,
        "labels": labels,
    })

print("Validated image-mask-JSON pairs:", len(samples))


# --------------------------------------------------
# 3. Convert labels into a multi-label matrix
# --------------------------------------------------

# Each row represents an image.
# Each column represents a defect class.
# 1 = image contains that class
# 0 = image does not contain that class

all_classes = sorted({
    label
    for sample in samples
    for label in sample["labels"]
})

class_to_index = {
    label: index
    for index, label in enumerate(all_classes)
}

label_matrix = np.zeros(
    (len(samples), len(all_classes)),
    dtype=int
)

for row, sample in enumerate(samples):
    for label in sample["labels"]:
        label_matrix[row, class_to_index[label]] = 1

print("\nDefect classes:", all_classes)


# --------------------------------------------------
# 4. Split off 10% for the final test set
# --------------------------------------------------

indices = np.arange(len(samples)).reshape(-1, 1)

first_split = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.10,
    random_state=42
)

train_val_positions, test_positions = next(
    first_split.split(indices, label_matrix)
)


# --------------------------------------------------
# 5. Split the remaining data into train and validation
# --------------------------------------------------

# 10% of the original dataset is approximately
# 11.11% of the remaining 90%.

second_split = MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=1 / 9,
    random_state=42
)

train_relative, val_relative = next(
    second_split.split(
        indices[train_val_positions],
        label_matrix[train_val_positions]
    )
)

train_positions = train_val_positions[train_relative]
val_positions = train_val_positions[val_relative]

splits = {
    "train": train_positions,
    "validation": val_positions,
    "test": test_positions,
}


# --------------------------------------------------
# 6. Save split manifests as CSV files
# --------------------------------------------------

for split_name, positions in splits.items():
    output_path = SPLIT_DIR / f"{split_name}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "mask_filename",
                "json_filename",
                "labels",
            ],
        )

        writer.writeheader()

        for position in positions:
            sample = samples[int(position)]

            writer.writerow({
                "filename": sample["filename"],
                "mask_filename": sample["mask_filename"],
                "json_filename": sample["json_filename"],
                "labels": ";".join(sample["labels"]),
            })

    print(f"Saved {len(positions)} samples to {output_path}")


# --------------------------------------------------
# 7. Report class distribution in each split
# --------------------------------------------------

print("\n===== CLASS DISTRIBUTION =====")

for split_name, positions in splits.items():
    print(f"\n{split_name.upper()} ({len(positions)} images)")

    subset_labels = label_matrix[positions]

    for label, column in class_to_index.items():
        count = int(subset_labels[:, column].sum())
        print(f"  {label}: {count} images")


# --------------------------------------------------
# 8. Check that the splits do not overlap
# --------------------------------------------------

train_set = set(train_positions.tolist())
val_set = set(val_positions.tolist())
test_set = set(test_positions.tolist())

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

assert len(train_set | val_set | test_set) == len(samples)

print("\nSUCCESS: Splits do not overlap.")
print("SUCCESS: Every sample belongs to exactly one split.")