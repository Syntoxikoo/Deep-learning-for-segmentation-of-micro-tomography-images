import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ...models.Unet_ViT import UNetViT
from ...utils import (
    TRANSFORM,
    SIMPLE_TRANSFORM,
    DiceLoss,
    DiceMetric,
    TOMODataset,
    WeightCELoss,
    get_device,
    init_weights,
    plot_losses_curves,
    plot_prediction,
    setup_logger,
)

torch.cuda.empty_cache()

# Argument parser
parser = argparse.ArgumentParser(
    description="Train a U-Net ViT model for segmentation."
)
parser.add_argument("--img_data_path", type=str, default="datas/Unet/train/imgs")
parser.add_argument("--mask_data_path", type=str, default="datas/Unet/train/labels")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--save_dir", type=str, default="checkpoints")
parser.add_argument("--input_size", type=int, default=None)
parser.add_argument("--vit_num_layers", type=int, default=2)
parser.add_argument("--vit_num_heads", type=int, default=4)
parser.add_argument("--max_tokens", type=int, default=2048)
args = parser.parse_args()

timestamp = datetime.now().strftime("%d-%Hh%M")
save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
logger = setup_logger(save_dir)
csv_path = os.path.join(save_dir, "metrics.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "learning_rate"])
# ============ LOAD DATASETS ============
resize = [args.input_size, args.input_size] if args.input_size else None
transform = v2.Compose(list(SIMPLE_TRANSFORM.values()))
train_ds = TOMODataset(
    img_dir=args.img_data_path,
    label_dir=args.mask_data_path,
    split="train",
    resized_shape=resize,
    transform=transform,
)

val_ds = TOMODataset(
    img_dir=args.img_data_path,
    label_dir=args.mask_data_path,
    split="test",
    resized_shape=resize,
    transform=None,
)


train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")


# ============ DEVICE AND MODEL ============
device, dl_workers = get_device()

model = UNetViT(
    in_channels=1,
    num_classes=2,
    features=(64, 128, 256, 512),
    up_method="Ctranspose",
    normalize=True,
    filter_size=3,
    dropout=0.1,
    vit_num_layers=args.vit_num_layers,
    vit_num_heads=args.vit_num_heads,
    vit_mlp_dim=None,  # defaults to 4*channels
    vit_dropout=0.2,
    max_tokens=args.max_tokens,
    batchsize=args.batch_size,
).to(device)

model.apply(init_weights)
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")


# ============ OPTIMIZER AND SCHEDULER ============
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)
if device.type == "cuda":
    scaler = GradScaler()

dice_loss = DiceLoss()
wce_loss = WeightCELoss()
dice_metric = DiceMetric()


def combined_loss(pred, target, weights):
    return wce_loss(pred, target, weights) + dice_loss(pred, target)


# ============ TRAINING LOOP ============
best_dice = 0.0
wait = 0
stop_patience = 15
history_train_loss = []
history_val_loss = []
history_dice = []

for epoch in range(args.epochs):
    # Training Step
    model.train()
    epoch_losses = []

    for i, (images, labels, weights) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        optimizer.zero_grad()

        if device.type == "cuda":
            with autocast(device_type="cuda"):
                logits, deep_preds = model(images)
                loss = combined_loss(logits, labels, weights)

                # Add auxiliary loss from deep supervision
                for i, dp in enumerate(deep_preds):
                    aux_loss = combined_loss(dp, labels, weights)
                    loss += 0.5 * aux_loss  # Weight auxiliary losses

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits, deep_preds = model(images)
            loss = combined_loss(logits, labels, weights)

            # Add auxiliary loss from deep supervision
            for i, dp in enumerate(deep_preds):
                aux_loss = combined_loss(dp, labels, weights)
                loss += 0.5 * aux_loss  # Weight auxiliary losses

            loss.backward()
            optimizer.step()

        epoch_losses.append(loss.item())

        if i % 10 == 0:
            probs = torch.softmax(logits, dim=1)
            logger.info(
                f"[Epoch {epoch + 1}][Step {i}] Loss: {loss.item():.4f} | Class 0 Prob: {probs[:, 0].mean().item():.2f}"
            )

    avg_train_loss = np.mean(epoch_losses)

    history_train_loss.append(avg_train_loss)

    # -------- VALIDATION --------
    model.eval()
    val_loss = []
    val_dices = []

    with torch.no_grad():
        for images, labels, weights in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            # Forward pass
            logits, deep_preds = model(images)

            loss = combined_loss(logits, labels, weights)

            # Add auxiliary loss from deep supervision
            for dp in deep_preds:
                aux_loss = combined_loss(dp, labels, weights)
                loss += 0.5 * aux_loss

            val_loss.append(loss.item())

            pred = torch.softmax(logits, dim=1)
            dice = dice_metric(pred, labels)
            val_dices.append(dice)

    avg_val_loss = float(np.mean(val_loss))
    avg_val_dice = float(np.mean(val_dices))

    history_val_loss.append(avg_val_loss)
    history_dice.append(avg_val_dice)

    logger.info(f"=== Epoch {epoch + 1}/{args.epochs} Result ===")
    logger.info(f"Train Loss: {avg_train_loss:.4f}")
    logger.info(f"Test Loss:  {avg_val_loss:.4f}")
    logger.info(f"Dice Score: {avg_val_dice:.4f}")
    logger.info("===============================")

    # -------- EARLY STOPPING & MODEL SAVING --------
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        wait = 0
        best_model_path = os.path.join(save_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")
    else:
        wait += 1

    if wait >= stop_patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

    # Scheduler step
    scheduler.step(avg_val_dice)
    csv_writer.writerow(
        [
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            avg_val_dice,
            scheduler.get_last_lr()[0],
        ]
    )
    plot_prediction(
        images[0],
        labels[0],
        torch.argmax(logits, dim=1)[0],
        save_dir,
        epoch + 1,
        show=False,
    )


# ============ SAVE FINAL MODEL ============
final_model_path = os.path.join(args.save_dir, "last_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"✓ Final model saved to {final_model_path}")


# ============ SAVE TRAINING CURVES ============
plot_losses_curves(history_train_loss, history_val_loss, save_dir)

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)
