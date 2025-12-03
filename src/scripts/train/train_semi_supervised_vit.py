"""
Semi-Supervised Training with Mean Teacher and U-Net ViT.

1. Student network learns from labeled data (supervised loss)
2. Teacher network provides pseudo-labels for unlabeled data
3. Student learns to be consistent with teacher predictions (consistency loss)
4. Teacher weights are updated via exponential moving average (EMA)
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Add src folder to path
REPO_ROOT = Path(__file__).resolve().parents[3]
print(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:  
    sys.path.insert(0, str(REPO_ROOT))

from src.models.mean_teacher_Unet import MeanTeacherUNet
from src.utils import (
    TRANSFORM,
    AddGaussianNoise,
    ConsistencyLoss,
    DiceLoss,
    DiceMetric,
    TOMODataset,
    WeightCELoss,
    get_device,
    init_weights,
    plot_losses_curves,
    plot_prediction,
    plot_prediction_v2,
    setup_logger,
    visualize_student_vs_teacher,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Semi-Supervised Mean Teacher U-Net ViT model."
    )
    # Data paths
    parser.add_argument(
        "--img_dir",
        type=str,
        default="datas/Original Images",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default="datas/Original Masks",
    )
    parser.add_argument(
        "--unlabeled_dir",
        type=str,
        default="datas/1h_HT",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
    )
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=512)

    # Semi-supervised hyperparameters
    parser.add_argument(
        "--ema_alpha", type=float, default=0.99, help="EMA decay for teacher"
    )
    parser.add_argument(
        "--lambda_u",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.6,
    )

    # Model hyperparameters
    parser.add_argument("--features", type=str, default="64,128,256,512")
    parser.add_argument("--vit_num_layers", type=int, default=2)
    parser.add_argument("--vit_num_heads", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Checkpoints
    parser.add_argument(
        "--student_ckpt", type=str, default=None, help="Path to student checkpoint"
    )
    parser.add_argument(
        "--teacher_ckpt", type=str, default=None, help="Path to teacher checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ============ SETUP ============
    timestamp = datetime.now().strftime("%d-%Hh%M")
    save_dir = os.path.join(args.save_dir, f"semi_vit_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(save_dir)
    logger.info(f"Arguments: {args}")

    # CSV logging
    csv_path = os.path.join(save_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["epoch", "sup_loss", "cons_loss", "val_loss", "val_dice", "lambda_w", "lr"]
    )

    device, dl_workers = get_device()
    logger.info(f"Using device: {device}")

    # ============ DATASETS ============
    resize = (args.input_size, args.input_size)

    transform = v2.Compose(list(TRANSFORM.values()))
    # Labeled dataset (with masks)
    train_ds = TOMODataset(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split="train",
        resized_shape=resize,
        transform=transform,
    )

    val_ds = TOMODataset(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split="test",
        resized_shape=resize,
        transform=None,
    )

    # Unlabeled dataset

    # Unlabeled preprocessing: center crop
    # unlabeled_crop = v2.CenterCrop((512, 512))
    # unlabeled_crop = v2.CenterCrop((750, 750))

    # unlabeled_transform = v2.Compose([
    #     unlabeled_crop,               # remove circular border
    #     *list(TRANSFORM.values()),    # same augmentations as labeled dataset
    # ])
    unlabeled_transform = transform

    # #  Preview of CENTER CROP (before augmentation)
    # # load a raw unlabeled image (first one in the folder)
    # valid_ext = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    # image_files = [f for f in os.listdir(args.unlabeled_dir) if f.lower().endswith(valid_ext)]
    # raw_path = os.path.join(args.unlabeled_dir, image_files[0])
    # # Load TIFF correctly (16-bit safe)
    # raw_img = Image.open(raw_path)
    # raw_np = np.array(raw_img).astype(float)
    # # Normalize to [0,1] for visualization
    # raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8)
    # print("Raw TIFF array range:", raw_np.min(), raw_np.max())
    # print("Raw TIFF shape:", raw_np.shape)
    # # Center crop manually (750×750 here)
    # crop_h, crop_w = 750, 750
    # H, W = raw_np.shape
    # y0 = (H - crop_h) // 2
    # x0 = (W - crop_w) // 2
    # cropped_np = raw_np[y0:y0+crop_h, x0:x0+crop_w]
    # plt.figure(figsize=(5,5))
    # plt.imshow(cropped_np, cmap="gray")
    # plt.title("Raw unlabeled TIFF after CENTER CROP (pre-augmentation)")
    # plt.axis("off")
    # plt.savefig(os.path.join(save_dir, "unlabeled_crop_raw.png"))
    # plt.close()


    unlabeled_ds = TOMODataset(
        img_dir=args.unlabeled_dir,
        label_dir=None,
        split="train",
        train_ratio=1.0,
        resized_shape=resize,
        transform=unlabeled_transform,   # cropped images
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=dl_workers,
        pin_memory=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=dl_workers,
        pin_memory=True,
    )

    logger.info(
        f"Datasets: Train={len(train_ds)} | Val={len(val_ds)} | Unlabeled={len(unlabeled_ds)}"
    )

    # ============ MODEL ============
    features = tuple(int(x) for x in args.features.split(","))

    model = MeanTeacherUNet(
        in_channels=1,
        num_classes=2,
        features=features,
        normalize=True,
        drop_out=args.dropout,
        ema_alpha=args.ema_alpha,
        vit_num_layers=args.vit_num_layers,
        vit_num_heads=args.vit_num_heads,
        max_tokens=args.max_tokens,
        batchsize=args.batch_size,
    ).to(device)

    # Initialize weights
    model.student.apply(init_weights)
    model.teacher.load_state_dict(model.student.state_dict())

    # Load checkpoints if provided
    if args.student_ckpt:
        model.load_student(args.student_ckpt, device=device)  # type: ignore
    if args.teacher_ckpt:
        model.load_teacher(args.teacher_ckpt, device=device)  # type: ignore

    n_params = sum(p.numel() for p in model.student.parameters())
    logger.info(f"Model initialized. Student params: {n_params:,}")

    add_noise = AddGaussianNoise(0.03)

    # ============ OPTIMIZER & LOSSES ============
    optimizer = optim.AdamW(model.student.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    scaler = GradScaler() if device.type == "cuda" else None

    wce_loss_fn = WeightCELoss()
    dice_loss_fn = DiceLoss()
    cons_loss_fn = ConsistencyLoss(temperature=0.5, conf_thresh=args.conf_thresh)
    dice_metric = DiceMetric()

    def combined_sup_loss(pred, target, weights):
        return wce_loss_fn(pred, target, weights) + dice_loss_fn(pred, target)

    # ============ TRAINING LOOP ============
    best_dice = 0.0
    warmup_epochs = args.warmup_epochs or max(1, args.epochs // 2)
    unlabeled_iter = iter(unlabeled_loader)

    history = {
        "sup_loss": [],
        "cons_loss": [],
        "val_loss": [],
        "val_dice": [],
    }

    for epoch in range(args.epochs):
        model.train()
        model.student.train()
        model.teacher.eval()

        epoch_sup_losses = []
        epoch_cons_losses = []

        if epoch < 5:
            lambda_w = 0.0
        else:
            rampup_length = warmup_epochs - 5
            current_ramp = max(0.0, min(1.0, (epoch - 5) / rampup_length))
            lambda_w = float(args.lambda_u) * current_ramp

        for i, (imgs_l, labels_l, weights_l) in enumerate(train_loader):
            try:
                imgs_u = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                imgs_u = next(unlabeled_iter)

            # Move to device
            imgs_l = imgs_l.to(device, non_blocking=True)
            labels_l = labels_l.to(device, non_blocking=True)
            weights_l = weights_l.to(device, non_blocking=True)
            imgs_u = imgs_u.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast(device_type="cuda"):
                    # Supervised pass
                    stu_logits_l, deep_preds_l = model.student(imgs_l)
                    loss_sup = combined_sup_loss(stu_logits_l, labels_l, weights_l)

                    # Deep supervision - resize to match label size
                    for dp in deep_preds_l:
                        dp_resized = torch.nn.functional.interpolate(
                            dp,
                            size=labels_l.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        loss_sup += 0.5 * combined_sup_loss(
                            dp_resized, labels_l, weights_l
                        )

                    # Unsupervised pass
                    with torch.no_grad():
                        tea_logits_u, tea_deep_preds = model.teacher(imgs_u)

                    # imgs_u_noisy = add_noise(imgs_u)
                    imgs_u_noisy = imgs_u
                    stu_logits_u, stu_deep_preds = model.student(imgs_u_noisy)
                    loss_cons = cons_loss_fn(stu_logits_u, tea_logits_u)

                    for stu_dp, tea_dp in zip(stu_deep_preds, tea_deep_preds):
                        loss_cons += 0.5 * cons_loss_fn(stu_dp, tea_dp)

                    loss = loss_sup + lambda_w * loss_cons
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training (CPU/MPS)
                stu_logits_l, deep_preds_l = model.student(imgs_l)
                loss_sup = combined_sup_loss(stu_logits_l, labels_l, weights_l)

                # Deep supervision - resize to match label size
                for dp in deep_preds_l:
                    dp_resized = torch.nn.functional.interpolate(
                        dp,
                        size=labels_l.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss_sup += 0.5 * combined_sup_loss(dp_resized, labels_l, weights_l)

                with torch.no_grad():
                    tea_logits_u, _ = model.teacher(imgs_u)

                stu_logits_u, _ = model.student(imgs_u)
                loss_cons = cons_loss_fn(stu_logits_u, tea_logits_u)

                loss = loss_sup + lambda_w * loss_cons
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), max_norm=1.0)
                optimizer.step()

            model.update_teacher()

            epoch_sup_losses.append(loss_sup.item())
            epoch_cons_losses.append(loss_cons.item())

            if i % 10 == 0:
                logger.info(
                    f"[Epoch {epoch + 1}][Step {i}] "
                    f"Sup: {loss_sup.item():.4f} | "
                    f"Cons: {loss_cons.item():.4f} | "
                    f"Lambda: {lambda_w:.3f}"
                )

        avg_sup = np.mean(epoch_sup_losses)
        avg_cons = np.mean(epoch_cons_losses)
        history["sup_loss"].append(avg_sup)
        history["cons_loss"].append(avg_cons)

        # ============ VALIDATION ============
        model.eval()
        val_losses = []
        val_dices = []

        with torch.no_grad():
            for images, labels, weights in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)

                logits, deep_preds = model.student(images)
                loss = combined_sup_loss(logits, labels, weights)

                # Deep supervision - resize to match label size
                for dp in deep_preds:
                    dp_resized = torch.nn.functional.interpolate(
                        dp, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    loss += 0.5 * combined_sup_loss(dp_resized, labels, weights)

                val_losses.append(loss.item())
                dice = dice_metric(logits, labels)
                val_dices.append(dice)

        avg_val_loss = float(np.mean(val_losses))
        avg_val_dice = float(np.mean(val_dices))

        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)

        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        logger.info(f"Sup Loss: {avg_sup:.4f} | Cons Loss: {avg_cons:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        # ============ CHECKPOINTING ============
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(
                model.student.state_dict(),
                os.path.join(save_dir, "best_student.pth"),
            )
            torch.save(
                model.teacher.state_dict(),
                os.path.join(save_dir, "best_teacher.pth"),
            )
            torch.save(
                {
                    "student": model.student.state_dict(),
                    "teacher": model.teacher.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                    "args": vars(args),
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            logger.info(f"Saved new best model (Dice: {best_dice:.4f})")

            # Save prediction visualization
            visualize_student_vs_teacher(model, unlabeled_loader, device, save_dir)
            plot_prediction(
            #plot_prediction_v2(
                images[0],
                labels[0],
                torch.argmax(logits, dim=1)[0],
                save_dir,
                epoch + 1,
                show=False,
            )

        # Scheduler step
        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]["lr"]

        csv_writer.writerow(
            [
                epoch + 1,
                avg_sup,
                avg_cons,
                avg_val_loss,
                avg_val_dice,
                lambda_w,
                current_lr,
            ]
        )
        csv_file.flush()

    # ============ FINAL SAVE ============
    csv_file.close()
    torch.save(model.student.state_dict(), os.path.join(save_dir, "last_student.pth"))
    torch.save(model.teacher.state_dict(), os.path.join(save_dir, "last_teacher.pth"))

    # Plot training curves
    plot_losses_curves(history["sup_loss"], history["val_loss"], save_dir)

    logger.info(f"Training completed. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
