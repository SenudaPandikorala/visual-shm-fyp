
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Paths
PROJECT_PATH = Path(__file__).resolve().parents[1]

DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/"
    "Reinforced concrete structure segmentation dataset"
)

IMAGE_DIR = DATASET_PATH / "imagedata"
MASK_DIR = DATASET_PATH / "maskdata"
SPLIT_DIR = PROJECT_PATH / "splits"

# All images and masks will have these dimensions.
IMAGE_SIZE = 256

# Six classes, including background.
NUM_CLASSES = 6


class StructuralDefectDataset(Dataset):

    def __init__(self, split_csv):
        # Read the CSV describing the selected data split.
        self.data = pd.read_csv(split_csv)

    def __len__(self):
        # Number of samples in this split.
        return len(self.data)

    def __getitem__(self, index):
        # Get the filename for one sample.
        row = self.data.iloc[index]

        image_path = IMAGE_DIR / row["filename"]
        mask_path = MASK_DIR / row["mask_filename"]

        # Read the original image.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        # OpenCV loads colour images as BGR.
        # Convert BGR to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask without changing its integer class labels.
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        # Resize image and mask to the same fixed dimensions.
        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert image values from 0-255 to 0-1.
        image = image.astype(np.float32) / 255.0

        # Image: (height, width, channels)
        # PyTorch expects: (channels, height, width)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Mask: each pixel stores a class ID from 0 to 5.
        # CrossEntropyLoss requires integer class labels (long).
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask


if __name__ == "__main__":

    train_dataset = StructuralDefectDataset(
        SPLIT_DIR / "train.csv"
    )

    validation_dataset = StructuralDefectDataset(
        SPLIT_DIR / "validation.csv"
    )

    test_dataset = StructuralDefectDataset(
        SPLIT_DIR / "test.csv"
    )

    # Load a few samples at a time to manage memory.
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False
    )

    print("===== DATASET SIZES =====")
    print("Training:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))

    # Inspect one batch from the training set.
    images, masks = next(iter(train_loader))

    print("\n===== FIRST TRAINING BATCH =====")
    print("Image shape:", images.shape)
    print("Image data type:", images.dtype)
    print("Image minimum:", images.min().item())
    print("Image maximum:", images.max().item())

    print("\nMask shape:", masks.shape)
    print("Mask data type:", masks.dtype)
    print("Mask class IDs:", torch.unique(masks).tolist())