# %%
from pathlib import Path
import sys

# Add src folder
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
from torch import optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt

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

    # ========== Setup ==========
    path = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%d-%Hh%M")
    save_dir = os.path.join(path, "checkpoints", f"mean_teacher_s1_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    csv_file = open(os.path.join(save_dir, "metrics.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "sup_loss", "cons_loss", "val_loss", "val_dice", "lambda_w"])

    device, dl_workers = get_device()

    # ========== Augmentations ==========
    transform = v2.Compose(
        [
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
        ]
    )
    pair_transform = PairTransform(transform)

    # ========== Paths ==========
    if data_dir is None:
        data_dir = os.path.join(path, "datas")
    if unlabeled_dir is None:
        unlabeled_dir = os.path.join(data_dir, "1h_HT")

    img_dir = os.path.join(data_dir, "Original Images")
    label_dir = os.path.join(data_dir, "Original Masks")

    # ========== Datasets ==========
    resized_shape = (512, 512)

    train_dataset = TOMODataset(
        img_dir,
        label_dir,
        split="train",
        transform=pair_transform,
        resized_shape=resized_shape,
    )

    val_dataset = TOMODataset(
        img_dir,
        label_dir,
        split="test",
        resized_shape=resized_shape,
    )

    # unlabeled set gets *same* image preprocessing (normalize + resize),
    # the only difference is that there is no mask / Otsu step.
    unl_dataset = UnlabeledTomoDataset(
        unlabeled_dir,
        resized_shape=resized_shape,
    )

    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=dl_workers)
    val_loader = DataLoader(val_dataset, batch_size, True, num_workers=dl_workers)
    unl_loader = DataLoader(unl_dataset, batch_size, True, num_workers=dl_workers)

    # ========== Model ==========
    model = MeanTeacherUNet(
        in_channels=1,
        num_classes=2,
        ema_alpha=ema_alpha,
    ).to(device)

    sup_loss_fn = WeightedCrossEntropyLossV2()
    cons_loss_fn = ConsistencyLoss(temperature=0.5, conf_thresh=0.6)

    optimizer = optim.AdamW(model.student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler() if on_cluster else None

    best_val_loss = float("inf")
    unl_iter = iter(unl_loader)

    # warm-up for unsupervised term over first half of training
    warmup_epochs = max(1, epochs // 2)

    # ========== Training Loop ==========
    for epoch in range(epochs):
        model.train()
        model.student.train()
        model.teacher.eval()

        # epoch-level supervised / consistency losses
        sup_losses = []
        cons_losses = []

        # current lambda weight (after warm-up)
        lambda_w = float(lambda_u) * min(1.0, (epoch + 1) / warmup_epochs)

        for imgs_l, labels_l, weights_l in train_loader:
            # Get unlabeled batch
            try:
                imgs_u, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unl_loader)
                imgs_u, _ = next(unl_iter)

            imgs_l = imgs_l.to(device)
            labels_l = labels_l.to(device)
            weights_l = weights_l.to(device)
            imgs_u = imgs_u.to(device)

            optimizer.zero_grad(set_to_none=True)

            # ----- SUPERVISED PASS -----
            stu_logits_l = model.student(imgs_l)
            loss_sup = sup_loss_fn(stu_logits_l, labels_l, weights_l)
            loss_sup.backward()

            # ----- UNSUPERVISED PASS -----
            with torch.no_grad():
                tea_logits_u = model.teacher(imgs_u)

            stu_logits_u = model.student(imgs_u)
            loss_cons = cons_loss_fn(stu_logits_u, tea_logits_u)
            loss_u = lambda_w * loss_cons
            loss_u.backward()

            optimizer.step()
            model.update_teacher()

            sup_losses.append(loss_sup.item())
            cons_losses.append(loss_cons.item())

        scheduler.step()

        # ========== Validation ==========
        model.eval()
        val_losses = []
        dices = []

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
                dice = (2 * inter) / (union + 1e-8)
                dices.append(dice.item())

        avg_sup = float(np.mean(sup_losses))
        avg_cons = float(np.mean(cons_losses))
        avg_val = float(np.mean(val_losses))
        avg_dice = float(np.mean(dices))

        # no greek chars -> no UnicodeEncodeError on Windows terminals
        logger.info(
            f"Epoch {epoch+1}: SUP={avg_sup:.4f} CONS={avg_cons:.4f} "
            f"lam_w={lambda_w:.3f} VAL={avg_val:.4f} DICE={avg_dice:.4f}"
        )

        csv_writer.writerow([epoch + 1, avg_sup, avg_cons, avg_val, avg_dice, lambda_w])

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.student.state_dict(), os.path.join(save_dir, "best_student.pth"))
            torch.save(model.teacher.state_dict(), os.path.join(save_dir, "best_teacher.pth"))

    csv_file.close()

    # ========== VISUALIZATION (student vs teacher, + unlabeled) ==========
    print("\n========== VISUALIZATION (S1) ==========\n")
    best_student = os.path.join(save_dir, "best_student.pth")
    best_teacher = os.path.join(save_dir, "best_teacher.pth")
    if not (os.path.exists(best_student) and os.path.exists(best_teacher)):
        print("No best_* checkpoints found, skipping visualization.")
        return

    print(
        "Loading best models from:\n  ",
        best_student,
        "\n  ",
        best_teacher,
    )

    model.student.load_state_dict(torch.load(best_student, map_location=device))
    model.teacher.load_state_dict(torch.load(best_teacher, map_location=device))

    model.eval()
    model.student.eval()
    model.teacher.eval()

    # ---- Labeled examples ----
    print("\nLabeled test samples (image / GT / student / teacher)\n")
    with torch.no_grad():
        for i, (images, labels, _) in enumerate(val_loader):
            if i >= 2:
                break

            images = images.to(device)
            labels = labels.to(device)

            stu_out = model.student(images)
            tea_out = model.teacher(images)

            stu_pred = torch.argmax(stu_out, dim=1).cpu().numpy()[0]
            tea_pred = torch.argmax(tea_out, dim=1).cpu().numpy()[0]
            img_np = images.cpu().numpy()[0, 0]
            gt_np = labels.cpu().numpy()[0]

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(gt_np, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(stu_pred, cmap="gray")
            axes[2].set_title("Student Prediction")
            axes[2].axis("off")

            axes[3].imshow(tea_pred, cmap="gray")
            axes[3].set_title("Teacher Prediction")
            axes[3].axis("off")

            plt.tight_layout()
            plt.show()

    # ---- Unlabeled examples ----
    print("\nUnlabeled samples (image / student / teacher)\n")
    unl_loader_vis = DataLoader(unl_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for i, (images_u, _) in enumerate(unl_loader_vis):
            if i >= 2:
                break

            images_u = images_u.to(device)
            stu_out = model.student(images_u)
            tea_out = model.teacher(images_u)

            stu_pred = torch.argmax(stu_out, dim=1).cpu().numpy()[0]
            tea_pred = torch.argmax(tea_out, dim=1).cpu().numpy()[0]
            img_np = images_u.cpu().numpy()[0, 0]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("Unlabeled Image")
            axes[0].axis("off")

            axes[1].imshow(stu_pred, cmap="gray")
            axes[1].set_title("Student Prediction")
            axes[1].axis("off")

            axes[2].imshow(tea_pred, cmap="gray")
            axes[2].set_title("Teacher Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()


def _arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_cluster", type=lambda x: x.lower() in ["true", "1", "yes"], default=False)
    parser.add_argument("--batch_size", type=int, default=1)
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
