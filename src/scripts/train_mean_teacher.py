#%%
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

import argparse
import torch
import os
import csv
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch import optim, nn
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from utils import (
    get_device,
    PairTransform,
    TOMODataset,
    WeightedCrossEntropyLossV2,
    setup_logger,
)
from utils.unlabeled_loader import UnlabeledTomoDataset
from models.mean_teacher_Unet import MeanTeacherUNet, ConsistencyLoss


def main(
    on_cluster=True,
    batch_size=1,
    epochs=10,
    learning_rate=1e-4,
    data_dir=None,
    unlabeled_dir=None,
    ema_alpha=0.99,
    lambda_u=0.5,
):

    path = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%d-%Hh%M")
    save_dir = os.path.join(path, "checkpoints", f"mean_teacher_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    csv_file = open(os.path.join(save_dir, "metrics.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "sup_loss", "cons_loss", "val_loss", "val_dice"]
    )

    device, dl_workers = get_device()

    # transforms
    transform = v2.Compose([
        v2.RandomRotation([-20, 20]),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(
            degrees=180,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
        ),
        v2.ElasticTransform(alpha=50, sigma=5),
        v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    pair_transform = PairTransform(transform)

    if data_dir is None:
        data_dir = os.path.join(path, "datas")
    if unlabeled_dir is None:
        unlabeled_dir = os.path.join(data_dir, "1h_HT")

    img_dir   = os.path.join(data_dir, "Original Images")
    label_dir = os.path.join(data_dir, "Original Masks")

    train_dataset = TOMODataset(img_dir, label_dir, split="train", transform=pair_transform)
    val_dataset   = TOMODataset(img_dir, label_dir, split="test")
    unl_dataset   = UnlabeledTomoDataset(unlabeled_dir)

    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=dl_workers)
    val_loader   = DataLoader(val_dataset,   batch_size, True, num_workers=dl_workers)
    unl_loader   = DataLoader(unl_dataset,   batch_size, True, num_workers=dl_workers)

    model = MeanTeacherUNet(
        in_channels=1,
        num_classes=2,
        ema_alpha=ema_alpha,
    ).to(device)

    sup_loss_fn  = WeightedCrossEntropyLossV2()
    cons_loss_fn = ConsistencyLoss()

    optimizer = optim.AdamW(model.student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler() if on_cluster else None

    unl_iter = iter(unl_loader)
    best_val_loss = float("inf")

    # training loop
    for epoch in range(epochs):
        model.train()

        sup_losses  = []
        cons_losses = []

        for images, labels, weights in train_loader:

            try:
                images_u, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unl_loader)
                images_u, _ = next(unl_iter)

            images   = images.to(device)
            labels   = labels.to(device)
            weights  = weights.to(device)
            images_u = images_u.to(device)

            optimizer.zero_grad()

            if on_cluster:
                with autocast(device_type="cuda"):

                    stu_logits = model.student(images)
                    loss_sup = sup_loss_fn(stu_logits, labels, weights)

                    stu_u_logits = model.student(images_u)
                    with torch.no_grad():
                        tea_u_logits = model.teacher(images_u)

                    loss_cons = cons_loss_fn(stu_u_logits, tea_u_logits)

                    loss = loss_sup + lambda_u * loss_cons

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                stu_logits = model.student(images)
                loss_sup = sup_loss_fn(stu_logits, labels, weights)

                stu_u_logits = model.student(images_u)
                with torch.no_grad():
                    tea_u_logits = model.teacher(images_u)

                loss_cons = cons_loss_fn(stu_u_logits, tea_u_logits)

                loss = loss_sup + lambda_u * loss_cons

                loss.backward()
                optimizer.step()

            model.update_teacher()

            sup_losses.append(loss_sup.item())
            cons_losses.append(loss_cons.item())

        scheduler.step()

        # validation
        model.eval()
        val_losses = []
        dice_scores = []

        with torch.no_grad():
            for images, labels, weights in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                out = model.student(images)
                loss = sup_loss_fn(out, labels, weights)
                val_losses.append(loss.item())

                pred = torch.argmax(out, dim=1)
                inter = ((pred == 0) & (labels == 0)).sum()
                union = (pred == 0).sum() + (labels == 0).sum()

                dice_scores.append((2 * inter / (union + 1e-8)).item())

        avg_val_loss = np.mean(val_losses)
        avg_dice = np.mean(dice_scores)

        logger.info(
            f"Epoch {epoch+1} | Sup: {np.mean(sup_losses):.4f} | "
            f"Cons: {np.mean(cons_losses):.4f} | Val: {avg_val_loss:.4f} | Dice: {avg_dice:.4f}"
        )

        csv_writer.writerow([
            epoch+1,
            np.mean(sup_losses),
            np.mean(cons_losses),
            avg_val_loss,
            avg_dice
        ])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.student.state_dict(), os.path.join(save_dir, "best_student.pth"))
            torch.save(model.teacher.state_dict(), os.path.join(save_dir, "best_teacher.pth"))


    csv_file.close()


def _arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_cluster", type=lambda x: x.lower() in ["true","1","yes"], default=False)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--unlabeled_dir", type=str, default=None)
    parser.add_argument("--ema_alpha", type=float, default=0.99)
    parser.add_argument("--lambda_u", type=float, default=0.5)
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = _arg_parse()
    main(**vars(args))
