
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import StructuralDefectDataset, SPLIT_DIR
from unet import UNet


PROJECT_PATH = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_PATH / "checkpoints" / "unet_best.pth"

NUM_CLASSES = 6

CLASS_NAMES = [
    "Background",
    "Crack",
    "Concrete spalling",
    "Rebar exposure",
    "Rebar corrosion",
    "Concrete crushing",
]

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

dataset = StructuralDefectDataset(
    SPLIT_DIR / "validation.csv"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

model = UNet(num_classes=NUM_CLASSES).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

true_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
pred_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
intersection = np.zeros(NUM_CLASSES, dtype=np.int64)

with torch.no_grad():
    for images, masks in loader:

        predictions = model(images.to(device)).argmax(dim=1).cpu()

        for class_id in range(NUM_CLASSES):
            true_pixels = masks == class_id
            pred_pixels = predictions == class_id

            true_counts[class_id] += true_pixels.sum().item()
            pred_counts[class_id] += pred_pixels.sum().item()
            intersection[class_id] += (
                true_pixels & pred_pixels
            ).sum().item()

print("\n===== VALIDATION PIXEL COUNTS =====")
print(f"{'Class':22s} {'Ground truth':>14s} {'Predicted':>14s} {'Overlap':>14s}")

for class_id, name in enumerate(CLASS_NAMES):
    print(
        f"{name:22s} "
        f"{true_counts[class_id]:14,d} "
        f"{pred_counts[class_id]:14,d} "
        f"{intersection[class_id]:14,d}"
    )

print("\n===== CLASSES PRESENT IN VALIDATION MASKS =====")

for class_id, name in enumerate(CLASS_NAMES):
    print(
        f"{name:22s}: "
        f"{'Present' if true_counts[class_id] > 0 else 'ABSENT'}"
    )

print("\n===== DIAGNOSTIC COMPLETE =====")