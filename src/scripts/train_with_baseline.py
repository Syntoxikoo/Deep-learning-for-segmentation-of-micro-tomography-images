from datetime import datetime
import torch
import os
import csv

from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from torchvision.transforms import v2
from torch import optim
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch import nn
from ..utils import (
    get_device,
    PairTransform,
    UnetDataset,
    init_weights,
    WeightedCrossEntropyLossV2,
    setup_logger,
)
from ..models import UNet


def main(on_cluster=True):
    # Setup directories
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f"metrics_run_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "train_loss", "val_loss", "val_dice", "learning_rate"]
    )

    device, dl_workers = get_device()

    # Constants
    BATCH_SIZE = 10
    NUM_EPOCH = 10
    LEARNING_RATE = 1e-4
    pad_image = None

    # Transforms
    transform = v2.Compose(
        [
            v2.RandomRotation([-20, 20]),
            v2.RandomVerticalFlip(),
            v2.RandomHorizontalFlip(),
            v2.ElasticTransform(interpolation=v2.InterpolationMode.NEAREST),
        ]
    )
    pair_transform = PairTransform(transform)

    # Load data
    cwd = Path.cwd().parent
    path = os.path.join(cwd, "datas/Unet")

    train_dataset = UnetDataset(
        path, split="train", transform=pair_transform, padding=pad_image
    )
    test_dataset = UnetDataset(
        path, split="test", transform=pair_transform, padding=pad_image
    )

    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        True,
        num_workers=dl_workers,
        pin_memory=True if on_cluster else False,
    )
    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
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
    )
    model.apply(init_weights)
    model.to(device)

    # Training Params
    loss_fn = WeightedCrossEntropyLossV2()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if on_cluster:
        scaler = GradScaler()

    best_test_loss = float("inf")
    # Training
    for epoch in range(NUM_EPOCH):

        # Training Step
        model.train()
        epoch_losses = []

        for i, (images, labels, weights) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad()

            if on_cluster:
                with autocast():
                    out = model(images)
                    loss = loss_fn(out, labels, weights)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

            else:
                out = model(images)
                loss = loss_fn(out, labels, weights)
                loss.backward()
                optimizer.step()

            epoch_losses.append(loss.item())

            if i % 10 == 0:
                probs = torch.softmax(out, dim=1)
                logger.info(
                    f"[Epoch {epoch+1}][Step {i}] Loss: {loss.item():.4f} | Class 0 Prob: {probs[:, 0].mean().item():.2f}"
                )

        avg_train_loss = np.mean(epoch_losses)
        scheduler.step()

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
                loss = loss_fn(out, labels, weights)
                test_loss_list.append(loss.item())

                prediction = torch.argmax(out, dim=1)

                # Calculating Dice for Class 0 (Background/Object?)
                # Ensure this matches your specific target class logic
                intersection = ((prediction == 0) & (labels == 0)).sum()
                union = (prediction == 0).sum() + (labels == 0).sum()
                dice = (2 * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())

        avg_test_loss = np.mean(test_loss_list)
        avg_dice = np.mean(dice_scores)

        logger.info(f"=== Epoch {epoch+1}/{NUM_EPOCH} Result ===")
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
        torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pth"))

        # Save Best
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            logger.info(f"New best model saved (Loss: {best_test_loss:.4f})")

    csv_file.close()


def show_prediction(images, labels, out, dir, epoch=0) -> None:
    images = images.cpu()
    labels = labels.cpu()
    preds = nn.functional.softmax(out, dim=1)
    preds = torch.argmax(preds, dim=1).cpu()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(images[0, 0], cmap="gray")
    axs[0].set_title("Input Image")
    axs[1].imshow(labels[0].numpy(), cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(preds[0].numpy(), cmap="gray")
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.savefig(os.path.join(dir, f"prediction_epoch_{epoch}.png"))


if __name__ == "__main__":
    main(False)
