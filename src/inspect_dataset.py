import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/Reinforced concrete structure segmentation dataset"
)
IMAGE_DIR = DATASET_PATH / "imagedata"
MASK_DIR = DATASET_PATH / "maskdata"
JSON_DIR = DATASET_PATH / "jsondata"

sample_name = "corrosion_2"
image_path = IMAGE_DIR / f"{sample_name}.jpg"
mask_path = MASK_DIR / f"{sample_name}.png"
json_path = JSON_DIR / f"{sample_name}.json"

#check if the files exist
print("Image exists:", image_path.exists())
print("Mask exists:", mask_path.exists())
print("JSON exists:", json_path.exists())

#using imageread to read the image and mask
image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

#printing dimensions
print("\n===== IMAGE =====")
print("Image shape:", image.shape)

print("\n===== MASK =====")
print("Mask shape:", mask.shape)
print("Mask data type:", mask.dtype)

#inspecting the unique values in the mask
unique_values = np.unique(mask)
print("Unique mask values:", unique_values)

with open(json_path, "r") as file:
    annotation_data = json.load(file)

print("\n===== JSON =====")
print("Image width:", annotation_data["imageWidth"])
print("Image height:", annotation_data["imageHeight"])
print("Number of annotations:", len(annotation_data["shapes"]))

print("\nLabels:")

for shape in annotation_data["shapes"]:
    print("-", shape["label"])


# Inspecting all dataset labels
print("\n===== ALL DATASET LABELS =====")

all_labels = set()

for json_file in JSON_DIR.glob("*.json"):
    with open(json_file, "r") as file:
        data = json.load(file)

    for shape in data["shapes"]:
        all_labels.add(shape["label"])

print("Labels found in dataset:")

for label in sorted(all_labels):
    print("-", label)


# Inspecting all mask values
print("\n===== ALL MASK VALUES =====")

all_mask_values = set()

for mask_file in MASK_DIR.glob("*.png"):
    current_mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

    if current_mask is not None:
        values = np.unique(current_mask)
        all_mask_values.update(values.tolist())

print("Mask values found in dataset:")
print(sorted(all_mask_values))

#to investigate the mapping between class IDs in the masks and the labels in the JSON files (to find what mask numbers actually mean)
print("\n===== CLASS ID INVESTIGATION =====")

class_mapping_candidates = {}

for json_file in JSON_DIR.glob("*.json"):

    with open(json_file, "r") as file:
        data = json.load(file)

    labels = {shape["label"] for shape in data["shapes"]}

    # We only want images containing one defect class
    if len(labels) == 1:

        label = next(iter(labels))
        sample = json_file.stem
        corresponding_mask = MASK_DIR / f"{sample}.png"

        if corresponding_mask.exists():

            current_mask = cv2.imread(
                str(corresponding_mask),
                cv2.IMREAD_UNCHANGED
            )

            values = np.unique(current_mask)

            # Remove background value 0
            defect_values = values[values != 0]

            if len(defect_values) == 1:
                class_id = int(defect_values[0])

                if label not in class_mapping_candidates:
                    class_mapping_candidates[label] = set()

                class_mapping_candidates[label].add(class_id)


for label, ids in sorted(class_mapping_candidates.items()):
    print(f"{label}: {sorted(ids)}")


# Convert OpenCV's BGR image to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image and segmentation mask
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(image_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

mask_display = axes[1].imshow(
    mask,
    cmap="tab10",
    vmin=0,
    vmax=5
)

axes[1].set_title("Ground-Truth Segmentation Mask")
axes[1].axis("off")

colorbar = plt.colorbar(
    mask_display,
    ax=axes[1],
    ticks=[0, 1, 2, 3, 4, 5]
)

colorbar.ax.set_yticklabels([
    "Background",
    "Crack",
    "Concrete spalling",
    "Rebar exposure",
    "Rebar corrosion",
    "Concrete crushing"
])

plt.tight_layout()
plt.show()