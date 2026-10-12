from pathlib import Path

import cv2
import torch


DATASET_PATH = Path(
    "/Users/senudapandikorala/Documents/FYP/Impl/Reinforced concrete structure segmentation dataset"
)

IMAGE_PATH = DATASET_PATH / "imagedata" / "corrosion_2.jpg"


# --------------------------------------------------
# 1. Load image using OpenCV
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

print("===== NUMPY IMAGE =====")
print("Type:", type(image))
print("Shape:", image.shape)
print("Data type:", image.dtype)


# --------------------------------------------------
# 2. Convert NumPy array to PyTorch tensor
# --------------------------------------------------

image_tensor = torch.from_numpy(image)

print("\n===== PYTORCH TENSOR =====")
print("Type:", type(image_tensor))
print("Shape:", image_tensor.shape)
print("Data type:", image_tensor.dtype)


# --------------------------------------------------
# 3. Change HWC -> CHW
# --------------------------------------------------

image_tensor = image_tensor.permute(2, 0, 1)

print("\n===== AFTER PERMUTE =====")
print("Shape:", image_tensor.shape)


# --------------------------------------------------
# 4. Convert pixel values to floating point
#    and scale 0-255 -> 0-1
# --------------------------------------------------

image_tensor = image_tensor.float() / 255.0

print("\n===== NORMALIZED TENSOR =====")
print("Data type:", image_tensor.dtype)
print("Minimum value:", image_tensor.min().item())
print("Maximum value:", image_tensor.max().item())


# --------------------------------------------------
# 5. Select processing device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("\n===== DEVICE =====")
print("Using:", device)


# --------------------------------------------------
# 6. Move tensor to device
# --------------------------------------------------

image_tensor = image_tensor.to(device)

print("Tensor device:", image_tensor.device)