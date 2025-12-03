import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch import optim

# import sys
# from pathlib import Path
# 
# # Add project root to sys.path so "src" becomes importable
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[3]  # go from .../scripts/train/ → project root
# 
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

from src.models.mean_teacher_Unet_v2 import MeanTeacherUNetV2
from src.utils.dataset_loader_vit import TOMODatasetViT  # NEW loader (same args)
from src.utils.metrics import DiceMetric
from src.utils.viz import plot_prediction
from src.utils.device import get_device
from src.utils.init import init_weights


# ---------------------------
# Loss functions
# ---------------------------
def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()

def bce_dice(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target) + dice_loss(pred, target)


def consistency_loss(stu_logits, tea_logits):
    stu = torch.sigmoid(stu_logits)
    tea = torch.sigmoid(tea_logits.detach())
    return F.mse_loss(stu, tea)


# ---------------------------
# Training
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="datas/Original Images")
    parser.add_argument("--label_dir", default="datas/Original Masks")
    parser.add_argument("--unlabeled_dir", default="datas/1h_HT")

    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_alpha", type=float, default=0.99)
    parser.add_argument("--lambda_u", type=float, default=10.0)
    parser.add_argument("--input_size", type=int, default=512)

    parser.add_argument("--student_ckpt", default=None)
    parser.add_argument("--teacher_ckpt", default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------
    # Device
    # -------------------------
    device, dl_workers = get_device()
    print("Using:", device)

    # -------------------------
    # Datasets (same args as before!)
    # -------------------------
    train_ds = TOMODatasetViT(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split="train",
        resized_shape=[args.input_size, args.input_size],
        transform=None  # new loader already applies correct preprocessing
    )

    val_ds = TOMODatasetViT(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        split="test",
        resized_shape=[args.input_size, args.input_size],
        transform=None
    )

    unlabeled_ds = TOMODatasetViT(
        img_dir=args.unlabeled_dir,
        label_dir=args.label_dir,   # loader needs the shape from masks but ignores values
        split=None,
        resized_shape=[args.input_size, args.input_size],
        transform=None
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=dl_workers)
    unl_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers)

    # -------------------------
    # Model
    # -------------------------
    model = MeanTeacherUNetV2(
        ema_alpha=args.ema_alpha,
        normalize=True
    ).to(device)

    model.student.apply(init_weights)
    model.teacher.load_state_dict(model.student.state_dict())

    if args.student_ckpt:
        model.load_student(args.student_ckpt, device)
    if args.teacher_ckpt:
        model.load_teacher(args.teacher_ckpt, device)

    optimizer = optim.AdamW(model.student.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    dice_metric = DiceMetric()

    best_dice = 0.0
    unl_it = iter(unl_loader)

    # -------------------------
    # Training Loop
    # -------------------------
    for epoch in range(args.epochs):
        model.train()
        model.student.train()
        model.teacher.eval()

        sup_losses, cons_losses = [], []

        for img_l, lab_l in train_loader:
            try:
                img_u, _ = next(unl_it)
            except StopIteration:
                unl_it = iter(unl_loader)
                img_u, _ = next(unl_it)

            img_l, lab_l = img_l.to(device), lab_l.to(device)
            img_u = img_u.to(device)

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                # ----- Supervised -----
                stu_logits_l, deep_l = model.student(img_l)
                loss_sup = bce_dice(stu_logits_l, lab_l.float())

                for d in deep_l:
                    d_rs = F.interpolate(d, size=lab_l.shape[-2:], mode="bilinear", align_corners=False)
                    loss_sup += 0.5 * bce_dice(d_rs, lab_l.float())

                # ----- Teacher (no grad) -----
                with torch.no_grad():
                    tea_logits_u, tea_deep = model.teacher(img_u)

                # ----- Student unsupervised -----
                stu_logits_u, stu_deep = model.student(img_u)

                loss_cons = consistency_loss(stu_logits_u, tea_logits_u)
                for sd, td in zip(stu_deep, tea_deep):
                    loss_cons += 0.5 * consistency_loss(sd, td)

                loss = loss_sup + args.lambda_u * loss_cons

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            model.update_teacher()

            sup_losses.append(loss_sup.item())
            cons_losses.append(loss_cons.item())

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        dices = []

        with torch.no_grad():
            for img_v, lab_v in val_loader:
                img_v, lab_v = img_v.to(device), lab_v.to(device)
                logits, _ = model.student(img_v)
                d = dice_metric(logits, lab_v)
                dices.append(d)

        avg_dice = float(np.mean(dices))
        print(f"[Epoch {epoch+1}] Dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.student.state_dict(), os.path.join(args.save_dir, "best_student_v2.pth"))
            torch.save(model.teacher.state_dict(), os.path.join(args.save_dir, "best_teacher_v2.pth"))
            print("Saved NEW BEST")

    print("Training complete. Best Dice =", best_dice)


if __name__ == "__main__":
    main()
