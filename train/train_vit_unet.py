import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
import numpy as np
import argparse # Import argparse

import torch
torch.cuda.empty_cache()


# Adjust the path to import U_net_ViT from the models directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))

from models.baseline_Unet_ViT import U_net_ViT

# Argument parsing
parser = argparse.ArgumentParser(description="Train a U-Net ViT model for segmentation.")
parser.add_argument("--img_data_path", type=str, default="../datas/Original Images",
                    help="Path to the directory containing original images.")
parser.add_argument("--mask_data_path", type=str, default="../datas/Original Masks",
                    help="Path to the directory containing mask images.")
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate for the optimizer.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Batch size for training.")
parser.add_argument("--save_dir", type=str, default="models/predicted_models",
                    help="Directory to save the trained model and predictions.")

args = parser.parse_args()
 
class MicroCTDataset(Dataset):
    """
    Loads TIFF images and masks using tifffile so 16-bit intensities
    are NOT destroyed (PIL breaks 16-bit TIFFs).
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".tif")
        ])

        self.mask_files = [
            f.replace(".tif", "").replace("image_v2_", "image_v2_mask_") + ".tif"
            for f in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # --- LOAD TIFF PROPERLY (16-bit OK) ---
        img = tiff.imread(img_path)
        mask = tiff.imread(mask_path)

        # If 3D → pick center slice (same as your testing code)
        if img.ndim == 3:
            img = img[img.shape[0] // 2]
        if mask.ndim == 3:
            mask = mask[mask.shape[0] // 2]

        # --- Convert to float32 in [0,1] ---
        img = img.astype("float32")
        img = img / img.max()   # normalize safely

        mask = mask.astype("float32")
        mask = (mask > 0).astype("float32")  # binary mask

        # --- Convert to tensors ---
        img = torch.from_numpy(img).unsqueeze(0)  # shape: [1,H,W]
        mask = torch.from_numpy(mask).unsqueeze(0)

        # --- Apply augmentations ---
        if self.transform:
            data = torch.cat([img, mask], dim=0)  # [2,H,W]
            data = self.transform(data)
            img, mask = data[0].unsqueeze(0), data[1].unsqueeze(0)

        return img, mask

# Define DiceLoss
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + smooth) / (union + smooth)

class AddGaussianNoise(object):
    def __init__(self, sigma=0.02):
        self.sigma = sigma

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.sigma
        return tensor + noise


# Configuration
img_dir = args.img_data_path
mask_dir = args.mask_data_path
batch_size = args.batch_size
num_epochs = args.epochs
learning_rate = args.lr

# Transformations
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=5, expand=True, fill=0),
    T.RandomResizedCrop((768, 768), scale=(0.9, 1.0)),
    AddGaussianNoise(sigma=0.02)
])

# Dataset and split
dataset = MicroCTDataset(img_dir, mask_dir, transform=train_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Disable augmentation on validation set
val_ds.dataset.transform = None

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = U_net_ViT(
    encode_in=(1, 64, 128, 256),
    encode_out=(64, 128, 256, 512),
    decode_in=(1024, 512, 256, 128),
    decode_out=(512, 256, 128, 64),
    normalize=True
).to(device)

# Loss and optimizer
criterion = lambda pred, target: (
    nn.BCEWithLogitsLoss()(pred, target) + DiceLoss()(pred, target)
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


best_dice = 0.0
patience = 20
wait = 0
train_losses = []
val_losses = []
dice_scores = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()
        pred = model(img)

        # Crop mask to match pred spatial size
        _, _, ph, pw = pred.shape
        mask_cropped = mask[:, :, :ph, :pw]

        loss = criterion(pred, mask_cropped)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    val_dice_scores = []
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)

            # Crop validation mask to prediction size
            _, _, ph, pw = pred.shape
            mask_cropped = mask[:, :, :ph, :pw]

            val_loss += criterion(pred, mask_cropped).item()

            # Calculate Dice score
            pred_sigmoid = torch.sigmoid(pred)
            pred_mask = (pred_sigmoid > 0.5).float()
            
            intersection = (pred_mask * mask_cropped).sum()
            union = pred_mask.sum() + mask_cropped.sum()
            dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
            val_dice_scores.append(dice_score.item())

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    avg_val_dice = np.mean(val_dice_scores)
    dice_scores.append(avg_val_dice)


    # Early stopping
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    scheduler.step(avg_val_dice)


    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

# Final average Dice score over all validation images 
final_avg_dice = np.mean(val_dice_scores) 
print(f"Final Average Dice Score over Validation Images: {final_avg_dice:.4f}")

# Save model
model_save_path = os.path.join(args.save_dir, "unet_vit_trained.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path} ✔")

# save loss plots
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(args.save_dir, "training_curves.png"))
plt.close()

plt.figure() 
plt.plot(dice_scores) 
plt.title("Validation Dice Score") 
plt.xlabel("Epoch") 
plt.ylabel("Dice") 
plt.savefig(os.path.join(args.save_dir, "dice_curves.png")) 
plt.close()

# Save an example prediction image
save_dir = os.path.join(args.save_dir, "predictions") # Subdirectory for predictions
os.makedirs(save_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    # Take one example from the validation set
    img_example, mask_example = val_ds[0]
    img_example = img_example.unsqueeze(0).to(device) # Add batch dimension

    pred_logits_example = model(img_example)
    pred_sigmoid_example = torch.sigmoid(pred_logits_example).cpu()
    pred_mask_example = (pred_sigmoid_example > 0.5).float()

    # Crop prediction to match original image size for saving consistency if needed
    _, _, oh, ow = img_example.shape
    _, _, ph, pw = pred_mask_example.shape

    if ph != oh or pw != ow:
        # If model output size differs, we might need to resize or handle cropping
        # For simplicity, let's assume the prediction is roughly centered or adjust as needed.
        # The notebook crops the mask to match prediction, here we save the prediction as is.
        pass

    # Save the prediction as a TIFF file
    pred_output_path = os.path.join(save_dir, "pred_example.tif")
    tiff.imwrite(pred_output_path, pred_mask_example.squeeze().numpy())
    gt_output_path = os.path.join(save_dir, "pred_ground_truth.tif")
    tiff.imwrite(gt_output_path, mask_example.squeeze().numpy())
    print(f"Example prediction and ground truth saved to {pred_output_path} ✔")