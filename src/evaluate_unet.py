
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from dataset import StructuralDefectDataset, SPLIT_DIR
from unet import UNet


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_PATH / "checkpoints" / "unet_best.pth"

NUM_CLASSES = 6
BATCH_SIZE = 4

CLASS_NAMES = [
    "Background",
    "Crack",
    "Concrete spalling",
    "Rebar exposure",
    "Rebar corrosion",
    "Concrete crushing",
]

# Consistent colours for ground-truth and predicted masks.
CLASS_COLORS = [
    "#000000",  # Background
    "#00FF00",  # Crack
    "#0000FF",  # Concrete spalling
    "#FF00FF",  # Rebar exposure
    "#FFFF00",  # Rebar corrosion
    "#FF0000",  # Concrete crushing
]

mask_cmap = ListedColormap(CLASS_COLORS)


# --------------------------------------------------
# 2. Select the device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------------------------------------
# 3. Load the validation dataset
# --------------------------------------------------

validation_dataset = StructuralDefectDataset(
    SPLIT_DIR / "validation.csv"
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Validation images:", len(validation_dataset))


# --------------------------------------------------
# 4. Load the trained model
# --------------------------------------------------

model = UNet(num_classes=NUM_CLASSES).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print(
    "Checkpoint validation loss:",
    f'{checkpoint["validation_loss"]:.4f}'
)


# --------------------------------------------------
# 5. Calculate the confusion matrix
# --------------------------------------------------

# Rows represent true classes.
# Columns represent predicted classes.
confusion_matrix = torch.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=torch.int64,
)

# Store a few images for visualization.
visual_samples = []

with torch.no_grad():

    for images, masks in validation_loader:

        images_device = images.to(device)

        # Output shape: [batch, 6, height, width]
        scores = model(images_device)

        # Select the highest-scoring class for each pixel.
        predictions = torch.argmax(scores, dim=1)

        predictions_cpu = predictions.cpu()
        masks_cpu = masks.cpu()

        # Accumulate pixel counts for every true/predicted
        # class combination.
        true_flat = masks_cpu.reshape(-1)
        pred_flat = predictions_cpu.reshape(-1)

        counts = torch.bincount(
            true_flat * NUM_CLASSES + pred_flat,
            minlength=NUM_CLASSES * NUM_CLASSES,
        )

        confusion_matrix += counts.reshape(
            NUM_CLASSES, NUM_CLASSES
        )

        # Save up to four examples for visual inspection.
        if len(visual_samples) < 4:
            for i in range(images.size(0)):
                if len(visual_samples) >= 4:
                    break

                visual_samples.append(
                    (
                        images[i].cpu(),
                        masks_cpu[i],
                        predictions_cpu[i],
                    )
                )


# --------------------------------------------------
# 6. Calculate IoU and Dice for each class
# --------------------------------------------------

cm = confusion_matrix.numpy()

iou_scores = []
dice_scores = []

print("\n===== PER-CLASS RESULTS =====")

for class_id, class_name in enumerate(CLASS_NAMES):

    true_positive = cm[class_id, class_id]

    false_positive = (
        cm[:, class_id].sum() - true_positive
    )

    false_negative = (
        cm[class_id, :].sum() - true_positive
    )

    union = true_positive + false_positive + false_negative

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if union == 0:
        iou = float("nan")
    else:
        iou = true_positive / union

    if dice_denominator == 0:
        dice = float("nan")
    else:
        dice = (
            2 * true_positive
            / dice_denominator
        )

    iou_scores.append(iou)
    dice_scores.append(dice)

    iou_text = "N/A" if np.isnan(iou) else f"{iou:.4f}"
    dice_text = "N/A" if np.isnan(dice) else f"{dice:.4f}"

    print(
        f"{class_name:22s} | "
        f"IoU: {iou_text:>6s} | "
        f"Dice: {dice_text:>6s}"
    )


# Calculate averages over the five defect classes.
# Background is excluded from these averages.
defect_ious = np.array(iou_scores[1:], dtype=float)
defect_dices = np.array(dice_scores[1:], dtype=float)

mean_iou = (
    np.nanmean(defect_ious)
    if not np.all(np.isnan(defect_ious))
    else float("nan")
)

mean_dice = (
    np.nanmean(defect_dices)
    if not np.all(np.isnan(defect_dices))
    else float("nan")
)

print("\n===== OVERALL DEFECT RESULTS =====")
print(f"Mean defect IoU:  {mean_iou:.4f}")
print(f"Mean defect Dice: {mean_dice:.4f}")


# --------------------------------------------------
# 7. Visualize predictions
# --------------------------------------------------

fig, axes = plt.subplots(
    len(visual_samples),
    3,
    figsize=(15, 4 * len(visual_samples)),
    squeeze=False,
)

for i, (image, true_mask, predicted_mask) in enumerate(
    visual_samples
):

    # Convert image from [channels, height, width]
    # into [height, width, channels].
    image_display = image.permute(1, 2, 0).numpy()

    axes[i, 0].imshow(image_display)
    axes[i, 0].set_title("Original Image")

    axes[i, 1].imshow(
        true_mask.numpy(),
        cmap=mask_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )
    axes[i, 1].set_title("Ground-Truth Mask")

    axes[i, 2].imshow(
        predicted_mask.numpy(),
        cmap=mask_cmap,
        vmin=0,
        vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )
    axes[i, 2].set_title("U-Net Prediction")

    for ax in axes[i]:
        ax.axis("off")

legend_handles = [
    plt.Rectangle(
        (0, 0), 1, 1,
        color=CLASS_COLORS[class_id],
        label=CLASS_NAMES[class_id],
    )
    for class_id in range(NUM_CLASSES)
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])

output_path = PROJECT_PATH / "validation_predictions.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

print("\nSaved visualization to:", output_path)

plt.show()