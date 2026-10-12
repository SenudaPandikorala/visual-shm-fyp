
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    StructuralDefectDataset,
    SPLIT_DIR,
)
from unet import UNet


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_CLASSES = 6

PROJECT_PATH = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_PATH / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "unet_best.pth"


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 3. Load the datasets
# --------------------------------------------------

train_dataset = StructuralDefectDataset(
    SPLIT_DIR / "train.csv"
)

validation_dataset = StructuralDefectDataset(
    SPLIT_DIR / "validation.csv"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# --------------------------------------------------
# 4. Create the model and training tools
# --------------------------------------------------

model = UNet(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 5. Training and validation
# --------------------------------------------------

best_validation_loss = float("inf")

for epoch in range(EPOCHS):

    # TRAINING
    model.train()

    training_loss_total = 0.0
    training_samples = 0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Predict class scores for every pixel.
        predictions = model(images)

        # Compare predictions against the correct masks.
        loss = criterion(predictions, masks)

        # Calculate gradients.
        loss.backward()

        # Update model weights.
        optimizer.step()

        training_loss_total += loss.item() * images.size(0)
        training_samples += images.size(0)

    training_loss = training_loss_total / training_samples

    # VALIDATION
    model.eval()

    validation_loss_total = 0.0
    validation_samples = 0

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)
            loss = criterion(predictions, masks)

            validation_loss_total += loss.item() * images.size(0)
            validation_samples += images.size(0)

    validation_loss = validation_loss_total / validation_samples

    # Report results for this epoch.
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Training Loss: {training_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )

    # Save the model with the lowest validation loss.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": validation_loss,
                "seed": SEED,
            },
            BEST_MODEL_PATH,
        )

        print("  Saved new best model.")

print("\n===== TRAINING COMPLETE =====")
print("Best validation loss:", f"{best_validation_loss:.4f}")
print("Best model saved to:", BEST_MODEL_PATH)