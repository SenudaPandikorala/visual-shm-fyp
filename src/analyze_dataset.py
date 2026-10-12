import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/Reinforced concrete structure segmentation dataset"
)

IMAGE_DIR = DATASET_PATH / "imagedata"
MASK_DIR = DATASET_PATH / "maskdata"
JSON_DIR = DATASET_PATH / "jsondata"


# --------------------------------------------------
# 1. COUNT FILES
# --------------------------------------------------

image_files = list(IMAGE_DIR.glob("*.jpg"))
mask_files = list(MASK_DIR.glob("*.png"))
json_files = list(JSON_DIR.glob("*.json"))

print("===== DATASET SIZE =====")
print("Images:", len(image_files))
print("Masks:", len(mask_files))
print("JSON files:", len(json_files))


# --------------------------------------------------
# 2. COUNT DEFECT CLASSES
# --------------------------------------------------

image_class_counts = Counter()
annotation_counts = Counter()

for json_file in json_files:

    with open(json_file, "r") as file:
        data = json.load(file)

    # Classes appearing in this image
    labels_in_image = set()

    for shape in data["shapes"]:
        label = shape["label"]

        annotation_counts[label] += 1
        labels_in_image.add(label)

    # Count an image only once for each class
    for label in labels_in_image:
        image_class_counts[label] += 1


print("\n===== IMAGES CONTAINING EACH CLASS =====")

for label, count in sorted(image_class_counts.items()):
    print(f"{label}: {count}")


print("\n===== TOTAL ANNOTATIONS =====")

for label, count in sorted(annotation_counts.items()):
    print(f"{label}: {count}")



print("\n===== NUMBER OF DEFECT CLASSES PER IMAGE =====")

classes_per_image = Counter()

for json_file in json_files:

    with open(json_file, "r") as file:
        data = json.load(file)

    unique_labels = {
        shape["label"]
        for shape in data["shapes"]
    }

    classes_per_image[len(unique_labels)] += 1


for number_of_classes, count in sorted(classes_per_image.items()):
    print(
        f"{number_of_classes} defect class(es): "
        f"{count} images"
    )

print("\n===== MULTI-DEFECT COMBINATIONS =====")

combinations = Counter()

for json_file in json_files:

    with open(json_file, "r") as file:
        data = json.load(file)

    labels = tuple(sorted({
        shape["label"]
        for shape in data["shapes"]
    }))

    combinations[labels] += 1


for combination, count in combinations.most_common():
    print(f"{combination}: {count}")


