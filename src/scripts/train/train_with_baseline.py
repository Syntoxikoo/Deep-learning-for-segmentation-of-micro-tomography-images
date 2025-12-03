import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ...models.baseline_Unet import UNet
from ...utils import (
    SIMPLE_TRANSFORM,
    TRANSFORM,
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


def main(
    on_cluster=True,
    batch_size=10,
    epochs=10,
    learning_rate=1e-4,
    data_dir=None,
    size=512,
    loss_choice: str = "both",
):
    # Setup directories

    path = Path(__file__).resolve().parents[3]
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
    transform = v2.Compose(list(SIMPLE_TRANSFORM.values()))

    # Load data
    if data_dir:
        data_path = data_dir
    else:
        data_path = os.path.join(path, "datas")
    img_dir = os.path.join(data_path, "Original Images")
    label_dir = os.path.join(data_path, "Original Masks")

    train_dataset = TOMODataset(
        img_dir,
        label_dir=label_dir,
        split="train",
        transform=transform,
        resized_shape=[size, size],
        gaussian_weight=True,
    )
    test_dataset = TOMODataset(
        img_dir,
        label_dir=label_dir,
        split="test",
        resized_shape=[size, size],
        gaussian_weight=True,
    )

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
        up_method="bilinear",
        normalize=True,
        drop_out=0.2,
        batch_size=batch_size,
    )
    model.apply(init_weights)
    model.to(device)

    # Training Params
    ce_loss_fn = WeightCELoss()
    dice_loss_fn = DiceLoss()

    def combined_loss(out, labels, weights):
        return ce_loss_fn(out, labels, weights) + dice_loss_fn(out, labels)

    if loss_choice == "both":
        loss_fn = combined_loss
    elif loss_choice == "dice":
        loss_fn = dice_loss_fn
    else:
        loss_fn = ce_loss_fn

    dice_metric = DiceMetric()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    if on_cluster:
        scaler = GradScaler()
    train_loss_list = []
    val_loss_list = []
    val_dice_list = []
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_losses.append(loss.item())

            if i % 10 == 0:
                probs = torch.softmax(out, dim=1)
                logger.info(
                    f"[Epoch {epoch + 1}][Step {i}] Loss: {loss.item():.4f} | Class 0 Prob: {probs[:, 0].mean().item():.2f}"
                )

        avg_train_loss = np.mean(epoch_losses)
        train_loss_list.append(avg_train_loss)

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

                # Compute Dice metric for class 1
                dice = dice_metric(out, labels)
                dice_scores.append(dice)

        avg_test_loss = np.mean(test_loss_list)
        avg_dice = np.mean(dice_scores)
        val_loss_list.append(avg_test_loss)
        val_dice_list.append(avg_dice)
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
        pred = torch.argmax(out, dim=1)[0]

        plot_prediction(images[0], labels[0], pred, save_dir, epoch + 1, show=False)
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

    plot_losses_curves(train_loss_list, val_loss_list, save_dir, show=False)
    csv_file.close()


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
    parser.add_argument("--loss", type=str, default="both")
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    args = _arg_parse()
    main(
        on_cluster=args.on_cluster,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        loss_choice=args.loss,
    )
