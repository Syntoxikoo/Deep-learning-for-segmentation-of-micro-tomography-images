# === train_unet_vit_semi.py ===
import os, sys, argparse, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------- repo path setup (same style as train_unet_vit.py) -----------------
REPO_ROOT = Path(__file__).resolve().parents[3]
print("REPO ROOT:", REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.dataset_loader_vit import TOMODatasetViT, UnlabeledTOMODatasetViT
from src.models.Unet_ViT_adapted_to_supervised_train import U_net_ViT
# === plotting & visualization (from plot_func.py) ===
from src.utils.plot_func import (
    plot_prediction,
    plot_losses_curves,
    visualize_student_vs_teacher
)

import albumentations as A

class UnlabeledWeakOnly:
    """
    Wraps a DataLoader that yields (weak, strong) images
    so that it yields only the weak view.
    """
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for img_w, img_s in self.loader:
            yield img_w  # return ONLY weak view

    def __len__(self):
        return len(self.loader)

class SimpleModelWrapper:
    def __init__(self, student, teacher):
        self.student = student
        self.teacher = teacher

    def eval(self):
        # Set both networks to eval mode
        self.student.eval()
        self.teacher.eval()

# Override argmax behavior by wrapping the model
class BinaryVisWrapper(SimpleModelWrapper):
    def eval(self):
        self.student.eval()
        self.teacher.eval()
    def __call__(self, x):
        # force 1 → 2 channels mimic for argmax
        s_final, _ = self.student(x)
        t_final, _ = self.teacher(x)

        s2 = torch.cat([-s_final, s_final], dim=1)
        t2 = torch.cat([-t_final, t_final], dim=1)

        return s2, t2
    
# ----------------- losses -----------------
class DiceLoss(nn.Module):
    def forward(self, p, t, smooth=1e-6):
        p = torch.sigmoid(p)
        return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


class DiceMetric:
    def __call__(self, p, t, eps=1e-6):
        # p is logits or probs; we threshold after sigmoid
        if p.dtype.is_floating_point:
            p = torch.sigmoid(p)
        p = (p.reshape(-1) > 0.5).float()
        t = t.reshape(-1)
        i = (p * t).sum()
        u = p.sum() + t.sum()
        return ((2 * i + eps) / (u + eps)).item()
class DiceMetricInv:
    def __call__(self, p, t, eps=1e-6):
        # p is logits or probs
        if p.dtype.is_floating_point:
            p = torch.sigmoid(p)

        # binarize prediction
        p = (p.reshape(-1) > 0.5).float()
        t = t.reshape(-1).float()

        # invert 1↔0 so that pore = foreground = 1
        p = 1 - p
        t = 1 - t

        # compute dice on inverted (pore) mask
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        return ((2 * intersection + eps) / (union + eps)).item()


dice_metric = DiceMetricInv()


# ----------------- mean-teacher EMA update -----------------
def update_teacher(student, teacher, alpha):
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data * (1.0 - alpha))


# ----------------- argument parsing -----------------
parser = argparse.ArgumentParser(description="Semi-supervised U-Net+ViT (Mean Teacher).")

parser.add_argument("--img_data_path", type=str, default="./datas/Original Images")
parser.add_argument("--mask_data_path", type=str, default=".//datas/Original Masks")
parser.add_argument("--unlabeled_path", type=str, default="./datas/10min_HT")

parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=1)
# parser.add_argument("--lambda_u", type=float, default=10.0, help="weight for unlabeled loss")
# parser.add_argument("--ema_alpha", type=float, default=0.99, help="EMA decay for teacher")
parser.add_argument("--lambda_u", type=float, default=1.0,
                    help="max weight for unlabeled loss (after warmup)")
parser.add_argument("--ema_alpha", type=float, default=0.99,
                    help="EMA decay for teacher after warmup")
# NEW: warmup for unsupervised loss
parser.add_argument("--warmup_epochs", type=int, default=5,
                    help="number of epochs to ramp up lambda_u")
# NEW: confidence threshold for teacher predictions
parser.add_argument("--conf_thresh", type=float, default=0.6,
                    help="teacher confidence threshold for unsupervised loss")


parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Optional supervised checkpoint to initialize both student & teacher")

parser.add_argument("--save_dir", type=str, default="./models/predicted_models_semi")

args = parser.parse_args()
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d-%H%M")
save_dir = os.path.join(args.save_dir, f"semi_vit_{timestamp}")
os.makedirs(save_dir, exist_ok=True)
print("Saving outputs to:", save_dir)

# ----------------- augmentations -----------------
# Labeled training augmentations (same as your current script)
train_transform_labeled = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=0, p=0.5),
    A.RandomResizedCrop((768, 768), scale=(0.85, 1.0), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
],
    additional_targets={"mask": "mask"}
)

# For unlabeled data, we define a weak and a strong pipeline
weak_transform_unlabeled = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
])

strong_transform_unlabeled = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.Rotate(limit=10, border_mode=0, p=0.5),
    # A.RandomResizedCrop((768, 768), scale=(0.85, 1.0), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

# ----------------- datasets & loaders -----------------
# Labeled
train_ds = TOMODatasetViT(
    img_dir=args.img_data_path,
    label_dir=args.mask_data_path,
    split="train",
    resized_shape=[768, 768],
    transform=train_transform_labeled
)

val_ds = TOMODatasetViT(
    img_dir=args.img_data_path,
    label_dir=args.mask_data_path,
    split="test",
    resized_shape=[768, 768],
    transform=None
)

# Unlabeled (center-cropped 768x768 before these transforms)
unlabeled_train_ds = UnlabeledTOMODatasetViT(
    img_dir=args.unlabeled_path,
    crop_size=(768, 768),
    weak_transform=weak_transform_unlabeled,
    strong_transform=strong_transform_unlabeled,
    split=None
)

print("Labeled train samples:", len(train_ds))
print("Labeled val samples:", len(val_ds))
print("Unlabeled train samples:", len(unlabeled_train_ds))

labeled_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
unlabeled_loader = DataLoader(unlabeled_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

# ----------------- models: student & teacher -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    return U_net_ViT(
        encode_in=(1, 64, 128, 256),
        encode_out=(64, 128, 256, 512),
        decode_in=(1024, 512, 256, 128),
        decode_out=(512, 256, 128, 64),
        normalize=True
    )

student = build_model().to(device)
teacher = build_model().to(device)

# Optionally load supervised weights into both
if args.pretrained_path is not None and os.path.isfile(args.pretrained_path):
    print(f"Loading pretrained weights from {args.pretrained_path}")
    state = torch.load(args.pretrained_path, map_location=device)
    student.load_state_dict(state, strict=False)
    msg = student.load_state_dict(state, strict=False)
    print(">>> WEIGHT LOADING:", msg)
    teacher.load_state_dict(state, strict=False)
else:
    # if no checkpoint, just copy random init from student to teacher once
    teacher.load_state_dict(student.state_dict(), strict=False)

# Teacher is not optimized directly
for p in teacher.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def supervised_loss(logits, target):
    return bce_loss(logits, target) + dice_loss(logits, target)

# def unsupervised_loss(student_logits, teacher_logits):
#     # MSE between probabilities (you can also try KL/BCE)
#     s_prob = torch.sigmoid(student_logits)
#     t_prob = torch.sigmoid(teacher_logits).detach()
#     return F.mse_loss(s_prob, t_prob)

# def unsupervised_loss(student_logits, teacher_logits, conf_thresh=0.6):
#     """
#     Mean-squared error on high-confidence teacher pixels only.
#     conf_thresh keeps pixels where t_prob > conf_thresh OR t_prob < 1-conf_thresh.
#     """

#     s_prob = torch.sigmoid(student_logits)
#     t_prob = torch.sigmoid(teacher_logits).detach()

#     # mask confident teacher predictions
#     high_conf = (t_prob > conf_thresh) | (t_prob < (1.0 - conf_thresh))
#     mask = high_conf.float()

#     mse = (s_prob - t_prob) ** 2
#     masked_mse = (mask * mse).sum() / (mask.sum() + 1e-6)

#     return masked_mse

def unsupervised_loss(student_logits, teacher_logits, conf_thresh=0.6):
    """
    Plain MSE between student and teacher probabilities.
    (conf_thresh is unused for now — we keep it in the signature for compatibility.)
    """
    s_prob = torch.sigmoid(student_logits)
    t_prob = torch.sigmoid(teacher_logits).detach()
    mse = (s_prob - t_prob) ** 2
    return mse.mean()


# ----------------- training loop -----------------
best_dice = 0.0
history_train_sup = []
history_train_unsup = []
history_val_loss = []
history_val_dice = []

stop_wait = 15
wait = 0

for epoch in range(args.epochs):
    student.train()
    teacher.eval()

    running_sup = 0.0
    running_unsup = 0.0

    # ----- compute lambda weight (unsupervised loss) with warmup -----
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lambda_weight = args.lambda_u * float(epoch + 1) / float(args.warmup_epochs)
    else:
        lambda_weight = args.lambda_u

    # ----- EMA schedule for teacher -----
    if epoch < 2:
        ema_alpha = 0.90
    elif epoch < 5:
        ema_alpha = 0.95
    else:
        ema_alpha = args.ema_alpha

    unlabeled_iter = iter(unlabeled_loader)

    for img_l, mask_l in labeled_loader:
        img_l = img_l.to(device)
        mask_l = mask_l.to(device).float()

        # ----- get a batch of unlabeled data -----
        try:
            img_u_w, img_u_s = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            img_u_w, img_u_s = next(unlabeled_iter)

        img_u_w = img_u_w.to(device)
        img_u_s = img_u_s.to(device)

        # ----- DEBUG VISUALIZATION OF UNLABELED INPUTS -----
        if epoch == 0:   # only show once

            # strong augmented image
            img_s_np = img_u_s[0].detach().cpu().numpy().squeeze()
            plt.figure(figsize=(6,6))
            plt.imshow(img_s_np, cmap="gray")
            plt.title("DEBUG: strong augmented unlabeled image")
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, "debug_strong_aug.png"))
            plt.close()

            # weak augmented image
            img_w_np = img_u_w[0].detach().cpu().numpy().squeeze()
            plt.figure(figsize=(6,6))
            plt.imshow(img_w_np, cmap="gray")
            plt.title("DEBUG: weak augmented unlabeled image")
            plt.colorbar()
            plt.savefig(os.path.join(save_dir, "debug_weak_aug.png"))
            plt.close()

            print("Saved debug_strong_aug.png and debug_weak_aug.png")

        # ----- supervised part -----
        optimizer.zero_grad()

        # out_l = student(img_l)
        # if isinstance(out_l, (tuple, list)):
        #     logits_l = out_l[0]
        # else:
        #     logits_l = out_l

        logits_l, _ = student(img_l)                         #changed too

        # ensure shapes match (like in your supervised script)
        if logits_l.dim() == 4 and mask_l.dim() == 4:
            Hp, Wp = logits_l.shape[2], logits_l.shape[3]
            Hm, Wm = mask_l.shape[2], mask_l.shape[3]
            h = min(Hp, Hm); w = min(Wp, Wm)
            if (h != Hm) or (w != Wm):
                mask_l = mask_l[:, :, :h, :w]
            if (h != Hp) or (w != Wp):
                logits_l = logits_l[:, :, :h, :w]

        sup_loss = supervised_loss(logits_l, mask_l)
        running_sup += sup_loss.item()

        # ----- unlabeled consistency part -----
        with torch.no_grad():
            # out_t = teacher(img_u_w)
            # if isinstance(out_t, (tuple, list)):
            #     logits_t = out_t[0]
            # else:
            #     logits_t = out_t
            # 
            #Teacher always returns (final_pred, deep_preds)
            logits_t, _ = teacher(img_u_w)


        # out_s = student(img_u_s)
        # if isinstance(out_s, (tuple, list)):
        #     logits_s = out_s[0]
        # else:
        #     logits_s = out_s

        noise = 0.2 * torch.randn_like(img_u_s)          #added noise to student input
        noisy_img = img_u_s + noise

        logits_s, _ = student(noisy_img)


        # safe crop
        if logits_s.dim() == 4 and logits_t.dim() == 4:
            Hu, Wu = logits_s.shape[2], logits_s.shape[3]
            Ht, Wt = logits_t.shape[2], logits_t.shape[3]
            h = min(Hu, Ht); w = min(Wu, Wt)
            if (h != Hu) or (w != Wu):
                logits_s = logits_s[:, :, :h, :w]
            if (h != Ht) or (w != Wt):
                logits_t = logits_t[:, :, :h, :w]

        # unsup_loss = unsupervised_loss(logits_s, logits_t)
        # running_unsup += unsup_loss.item()

        # loss = sup_loss + args.lambda_u * unsup_loss

        unsup_loss = unsupervised_loss(logits_s, logits_t, conf_thresh=args.conf_thresh)
        running_unsup += unsup_loss.item()

        loss = sup_loss + lambda_weight * unsup_loss

        loss.backward()
        optimizer.step()

        # EMA update of teacher after each step (scheduled alpha)
        update_teacher(student, teacher, ema_alpha)

    # ------------- validation on labeled val set (student) -------------
    student.eval()
    vrun = 0.0
    dices = []

    with torch.no_grad():
        for img_v, mask_v in val_loader:
            img_v = img_v.to(device)
            mask_v = mask_v.to(device).float()

            # out_v = student(img_v)
            # if isinstance(out_v, (tuple, list)):
            #     logits_v = out_v[0]
            # else:
            #     logits_v = out_v

            logits_v, _ = student(img_v)         #changed too


            if logits_v.dim() == 4 and mask_v.dim() == 4:
                Hp, Wp = logits_v.shape[2], logits_v.shape[3]
                Hm, Wm = mask_v.shape[2], mask_v.shape[3]
                h = min(Hp, Hm); w = min(Wp, Wm)
                if (h != Hm) or (w != Wm):
                    mask_v = mask_v[:, :, :h, :w]
                if (h != Hp) or (w != Wp):
                    logits_v = logits_v[:, :, :h, :w]

            loss_v = supervised_loss(logits_v, mask_v)
            vrun += loss_v.item()
            dices.append(dice_metric(logits_v, mask_v))

    avg_sup = running_sup / max(1, len(labeled_loader))
    avg_unsup = running_unsup / max(1, len(labeled_loader))
    avg_val_loss = vrun / max(1, len(val_loader))
    avg_val_dice = float(np.mean(dices)) if len(dices) else 0.0

    history_train_sup.append(avg_sup)
    history_train_unsup.append(avg_unsup)
    history_val_loss.append(avg_val_loss)
    history_val_dice.append(avg_val_dice)

    # print(f"Epoch {epoch+1}/{args.epochs} "
    #       f"| Sup: {avg_sup:.4f} "
    #       f"| Unsup: {avg_unsup:.4f} "
    #       f"| ValLoss: {avg_val_loss:.4f} "
    #       f"| ValDice: {avg_val_dice:.4f}")
    
    print(f"Epoch {epoch+1}/{args.epochs} "
      f"| Sup: {avg_sup:.4f} "
      f"| Unsup: {avg_unsup:.6f} "
      f"| λ_u: {lambda_weight:.3f} "
      f"| EMA: {ema_alpha:.3f} "
      f"| ValLoss: {avg_val_loss:.4f} "
      f"| ValDice: {avg_val_dice:.4f}")

    # early stopping based on val Dice
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        wait = 0

        # === SAVE CHECKPOINTS ===
        torch.save(student.state_dict(), os.path.join(save_dir, "best_student.pth"))
        torch.save(teacher.state_dict(), os.path.join(save_dir, "best_teacher.pth"))
        torch.save({
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "args": vars(args),
        }, os.path.join(save_dir, "best_model.pth"))

        print(f"Saved new BEST model (Dice={best_dice:.4f})")

        # === PLOT PREDICTIONS ON LABELED VALIDATION ===
        img0, mask0 = img_v[0], mask_v[0]
        pred_logits = logits_v
        plot_prediction(img0, mask0, torch.sigmoid(pred_logits)[0], save_dir, epoch+1)

        # === VISUALIZE STUDENT VS TEACHER ON UNLABELED ===
        weak_only_loader = UnlabeledWeakOnly(unlabeled_loader)
        # wrapper = SimpleModelWrapper(student, teacher)
        wrapper = BinaryVisWrapper(student, teacher)

        visualize_student_vs_teacher(
            model=wrapper,
            unlabeled_loader=weak_only_loader,
            device=device,
            save_dir=save_dir
        )
    else:
        wait += 1

    if wait >= stop_wait:
        print("Early stopping at epoch", epoch + 1)
        break

    scheduler.step(avg_val_dice)

torch.save(student.state_dict(), os.path.join(save_dir, "last_student.pth"))
torch.save(teacher.state_dict(), os.path.join(save_dir, "last_teacher.pth"))

# ----------------- plots -----------------
# === PLOT TRAINING CURVES ===
plot_losses_curves(
    train_losses=history_train_sup,
    val_losses=history_val_loss,
    save_dir=save_dir
)

plt.figure()
plt.plot(history_train_sup, label="train_sup")
plt.plot(history_train_unsup, label="train_unsup")
plt.plot(history_val_loss, label="val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "semi_training_curves.png"))
plt.close()

plt.figure()
plt.plot(history_val_dice, label="val_dice")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "semi_dice_curve.png"))
plt.close()

print("Semi-supervised training finished. Best Val Dice:", best_dice)

# run this to load pretrained weights:
# python src/scripts/train/train_semi_supervised_vit_v3.py --pretrained_path "./models/predicted_models/unet_vit_trained.pth" --epochs 4
