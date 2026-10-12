
import torch

from torch.utils.data import DataLoader
from dataset import StructuralDefectDataset, SPLIT_DIR
from unet import UNet


def main():

    # Select Apple's GPU when available.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # Load two training samples for this test.
    dataset = StructuralDefectDataset(
        SPLIT_DIR / "train.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    images, masks = next(iter(loader))

    # Create the model and move it to the selected device.
    model = UNet(num_classes=6).to(device)

    images = images.to(device)

    # Run a forward pass without calculating gradients.
    model.eval()

    with torch.no_grad():
        predictions = model(images)

    # Convert scores into predicted class IDs.
    predicted_masks = torch.argmax(predictions, dim=1)

    print("\n===== INPUT =====")
    print("Image shape:", images.shape)

    print("\n===== MODEL OUTPUT =====")
    print("Prediction scores shape:", predictions.shape)
    print("Predicted mask shape:", predicted_masks.shape)
    print("Predicted class IDs:",
          torch.unique(predicted_masks).cpu().tolist())

    print("\n===== MODEL CHECKS =====")
    print("All prediction scores finite:",
          torch.isfinite(predictions).all().item())

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Number of model parameters:", f"{parameter_count:,}")

    assert predictions.shape == (2, 6, 256, 256)
    assert predicted_masks.shape == (2, 256, 256)
    assert torch.isfinite(predictions).all().item()

    print("\nSUCCESS: U-Net forward-pass test passed!")


if __name__ == "__main__":
    main()