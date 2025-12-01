import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..models import UNet
from ..utils import (
    DiceLoss,
    PairTransform,
    TOMODataset,
    WeightedCrossEntropyLoss,
    WeightedCrossEntropyLossV2,
    get_device,
    init_weights,
    setup_logger,
)


def main(on_cluster=True, batch_size=10, epochs=10, learning_rate=1e-4, data_dir=None):
    # Setup directories

    path = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%d-%Hh%M")
    save_dir = os.path.join(path, "checkpoints", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(save_dir)
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "train_loss", "val_loss", "val_dice", "learning_rate"]
    )

    device, dl_workers = get_device()
    # Transforms
    transform = v2.Compose(
        [
            v2.RandomRotation([-20, 20]),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(
                degrees=180,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                interpolation=v2.InterpolationMode.BILINEAR,
            ),
            # v2.ElasticTransform(
            #     alpha=50,
            #     sigma=5,
            # ),
            # v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            # v2.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    )
    pair_transform = PairTransform(transform)

    # Load data
    if data_dir:
        data_path = data_dir
    else:
        data_path = os.path.join(path, "datas")
    img_dir = os.path.join(data_path, "Original Images")
    label_dir = os.path.join(data_path, "Original Masks")

    train_dataset = TOMODataset(
        img_dir, label_dir=label_dir, split="train", transform=pair_transform
    )
    test_dataset = TOMODataset(img_dir, label_dir=label_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        True,
        num_workers=dl_workers,
        pin_memory=True if on_cluster else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size,
        True,
        num_workers=dl_workers,
        pin_memory=True if on_cluster else False,
    )

    model = UNet(
        in_channels=1,
        num_classes=2,
        features=(64, 128, 256, 512),
        bilinear=False,
        normalize=True,
        drop_out=0.3,
        batch_size=batch_size,
    )
    model.apply(init_weights)
    model.to(device)

    # Training Params
    ce_loss_fn = WeightedCrossEntropyLossV2()
    dice_loss_fn = DiceLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    if on_cluster:
        scaler = GradScaler()

    best_test_loss = float("inf")
    # Training
    for epoch in range(epochs):
        # Training Step
        model.train()
        epoch_losses = []

        for i, (images, labels, weights) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad()

            if on_cluster:
                with autocast(device_type="cuda"):
                    out = model(images)
                    ce_loss = ce_loss_fn(out, labels, weights)
                    dice_loss = dice_loss_fn(out, labels)
                    loss = 5 * dice_loss + ce_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                out = model(images)
                ce_loss = ce_loss_fn(out, labels, weights)
                dice_loss = dice_loss_fn(out, labels)
                loss = 5 * dice_loss + ce_loss
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

            if i % 10 == 0:
                probs = torch.softmax(out, dim=1)
                logger.info(
                    f"[Epoch {epoch + 1}][Step {i}] Loss: {loss.item():.4f} | Class 0 Prob: {probs[:, 0].mean().item():.2f}"
                )

        avg_train_loss = np.mean(epoch_losses)

        # Validation Step
        model.eval()
        test_loss_list = []
        dice_scores = []

        with torch.no_grad():
            for images, labels, weights in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                out = model(images)
                ce_loss = ce_loss_fn(out, labels, weights)
                dice_loss = dice_loss_fn(out, labels)
                loss = 5 * dice_loss + ce_loss
                test_loss_list.append(loss.item())

                prediction = torch.argmax(out, dim=1)

                # Calculating Dice for Class 1
                intersection = ((prediction == 1) & (labels == 1)).sum()
                union = (prediction == 1).sum() + (labels == 1).sum()
                dice = (2 * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

        avg_test_loss = np.mean(test_loss_list)
        avg_dice = np.mean(dice_scores)
        scheduler.step(avg_test_loss)

        logger.info(f"=== Epoch {epoch + 1}/{epochs} Result ===")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Test Loss:  {avg_test_loss:.4f}")
        logger.info(f"Dice Score: {avg_dice:.4f}")
        logger.info("===============================")

        # Write to CSV
        csv_writer.writerow(
            [
                epoch + 1,
                avg_train_loss,
                avg_test_loss,
                avg_dice,
                scheduler.get_last_lr()[0],
            ]
        )
        show_prediction(images, labels, out, save_dir, epoch)

        # Save Last
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "last_model.pth"),
        )

        # Save Best
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth"),
            )
            logger.info(f"New best model saved (Loss: {best_test_loss:.4f})")

    csv_file.close()


def show_prediction(images, labels, out, dir, epoch=0) -> None:
    images = images.cpu()
    labels = labels.cpu()
    preds = torch.argmax(out, dim=1).cpu()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(images[0, 0], cmap="gray")
    axs[0].set_title("Input Image")

    axs[1].imshow(labels[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Ground Truth")

    axs[2].imshow(preds[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.savefig(os.path.join(dir, f"prediction_epoch_{epoch}.png"))
    plt.close()


def _arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--on_cluster",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument("--data_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _arg_parse()
    main(
        on_cluster=args.on_cluster,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
    )
