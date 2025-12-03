# train.py for training baseline unet with vit
# includes: deep supervision, dice over full test set, test plots, saving loss/Dice curves, eval example, no rotation-expand mismatch

# ======= TRAIN.PY =======
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from utils.dataset_loader_vit import TOMODatasetViT

from ...models.Unet_ViT import UNetViT
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

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
)
from Unet_ViT import U_net_ViT

parser = argparse.ArgumentParser(
    description="Train a U-Net ViT model for segmentation."
)
parser.add_argument(
    "--img_data_path", type=str, default="../../../datas/original/train/imgs"
)
parser.add_argument(
    "--mask_data_path", type=str, default="../../../datas/original/train/labels"
)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="../..models/predicted_models")
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
    resized_shape=[768, 768],  # Matches model input
    transform=train_transform,
)

val_ds = TOMODatasetViT(
    img_dir="datas/original/train/imgs",
    label_dir="datas/original/train/labels",
    split="test",
    resized_shape=[768, 768],
    transform=None,
)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)


# dataset=MicroCTDataset(args.img_data_path, args.mask_data_path, transform=train_transform)
# tbs=int(0.8*len(dataset)); vbs=len(dataset)-tbs
# train_ds,val_ds=random_split(dataset,[tbs,vbs]); val_ds.dataset.transform=None
# train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
# val_loader=DataLoader(val_ds,batch_size=1,shuffle=False)

model = UNetViT(
    in_channels=1,
    num_classes=2,
    features=(64, 128, 256, 512),
    up_method="bilinear",
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


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)
loss_fn = lambda p, t: nn.BCEWithLogitsLoss()(p, t) + DiceLoss()(p, t)

best_dice = 0
stop_wait = 15
history_losses = []
history_val = []
history_dice = []
wait = 0

for epoch in range(args.epochs):
    model.train()
    run = 0.0

    for img, mask in train_loader:
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        p = model(img)
        # unwrap if model returns (logits, aux)
        if isinstance(p, tuple) or isinstance(p, list):
            p = p[0]

        # ensure predictions dtype float and mask dtype float
        if not torch.is_floating_point(p):
            p = p.float()
        mask_f = mask.float()

        # safe: if the prediction and mask spatial shapes mismatch, crop to smallest common region
        # p: (B, C, Hp, Wp), mask_f: (B, C_mask, Hm, Wm) typically C=1
        if p.dim() == 4 and mask_f.dim() == 4:
            Hp, Wp = p.shape[2], p.shape[3]
            Hm, Wm = mask_f.shape[2], mask_f.shape[3]
            h = min(Hp, Hm)
            w = min(Wp, Wm)
            if (h != Hm) or (w != Wm):
                mask_f = mask_f[:, :, :h, :w]
            if (h != Hp) or (w != Wp):
                p = p[:, :, :h, :w]

        loss = loss_fn(p, mask_f)
        loss.backward()
        optimizer.step()

        run += loss.item()

    # validation
    model.eval()
    vrun = 0.0
    dices = []
    with torch.no_grad():
        for img, mask in val_loader:
            img = img.to(device)
            mask = mask.to(device)

            p = model(img)
            if isinstance(p, tuple) or isinstance(p, list):
                p = p[0]

            # ensure p float, mask float
            if not torch.is_floating_point(p):
                p = p.float()
            mask_f = mask.float()

            # safe overlap crop as above
            if p.dim() == 4 and mask_f.dim() == 4:
                Hp, Wp = p.shape[2], p.shape[3]
                Hm, Wm = mask_f.shape[2], mask_f.shape[3]
                h = min(Hp, Hm)
                w = min(Wp, Wm)
                if (h != Hm) or (w != Wm):
                    mask_f = mask_f[:, :, :h, :w]
                if (h != Hp) or (w != Wp):
                    p = p[:, :, :h, :w]

            vrun += loss_fn(p, mask_f).item()
            try:
                dices.append(dice_metric(p, mask_f))
            except Exception:
                # dice_metric expects logits or probs? it thresholds >0.5 internally.
                # convert logits -> probs for robustness
                probs = torch.sigmoid(p)
                dices.append(dice_metric(probs, mask_f))

    avg_loss = run / max(1, len(train_loader))
    history_losses.append(avg_loss)
    avg_val = vrun / max(1, len(val_loader))
    history_val.append(avg_val)
    avg_dice = float(np.mean(dices)) if len(dices) else 0.0
    history_dice.append(avg_dice)

    print(
        f"Epoch {epoch + 1}/{args.epochs} | Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Dice: {avg_dice:.4f}"
    )

    # early stopping logic
    if avg_dice > best_dice:
        best_dice = avg_dice
        wait = 0
        # save best model (optional)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "best_unet_vit.pth"))
    else:
        wait += 1

    if wait >= stop_wait:
        print("Early stopping epoch", epoch + 1)
        break

    # scheduler expecting a metric (you used mode='max')
    scheduler.step(avg_dice)

# final scheduler calls (if you still want them)
try:
    scheduler.step(avg_dice)
    scheduler.step(avg_dice)
except Exception:
    pass

# Save model
mp = os.path.join(args.save_dir, "unet_vit_trained.pth")
os.makedirs(args.save_dir, exist_ok=True)
torch.save(model.state_dict(), mp)
print("Model saved ✔", mp)

# Save loss plot
plt.figure()
plt.plot(history_losses, label="train")
plt.plot(history_val, label="val")
plt.grid(True)  # Show grid
plt.legend()
plt.xlabel("Epochs")  # X-axis label
plt.ylabel("Loss")  # Y-axis label
plt.title("Training and Validation Loss")  # Plot title
plt.savefig(os.path.join(args.save_dir, "training_curves.png"))
plt.close()
plt.figure()
plt.plot(history_dice, label="dice")
plt.legend()
plt.savefig(os.path.join(args.save_dir, "dice_curve.png"))
plt.close()


# One eval example plot from validation
# One eval example plot from validation
img_e, mask_e = val_ds[0]  # img_e, mask_e shapes: (1, H, W)
bimg = img_e.unsqueeze(0).to(device)  # -> (1, 1, H, W)
with torch.no_grad():
    out = model(bimg)
    out = out[0] if isinstance(out, (tuple, list)) else out
    pm = torch.sigmoid(out).cpu()
    # pm = torch.sigmoid(model(bimg)).cpu() # pm shape: (1, 1, Hp, Wp)  (maybe same H/W)

# select the single image prediction (H, W)
pm_vis = pm[0, 0]  # shape: (Hp, Wp) as a tensor

# convert all to 2D numpy arrays for plotting
img_vis = img_e.squeeze(0).cpu().numpy()  # now (H, W)
mask_vis = mask_e.squeeze(0).cpu().numpy()  # now (H, W)
pm_vis_np = pm_vis.detach().cpu().numpy()  # (Hp, Wp)

# If prediction size differs from GT due to safe-crop, resize/clip or show the overlapping region.
# Here we will crop/pad to the smallest common size to avoid shape mismatch:
h = min(img_vis.shape[0], pm_vis_np.shape[0], mask_vis.shape[0])
w = min(img_vis.shape[1], pm_vis_np.shape[1], mask_vis.shape[1])
img_vis = img_vis[:h, :w]
mask_vis = mask_vis[:h, :w]
pm_vis_np = pm_vis_np[:h, :w]

# compute dice on these CPU tensors (DiceMetric expects tensors)
dice_val = dice_metric(torch.from_numpy(pm_vis_np), torch.from_numpy(mask_vis))

# Plot input, GT, prediction
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_vis, cmap="gray", vmin=0.0, vmax=1.0)
plt.title("image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask_vis, cmap="gray")
plt.title("mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pm_vis_np, cmap="gray", vmin=0.0, vmax=1.0)
plt.title(f"pred dice {dice_val:.4f}")
plt.axis("off")

plt.tight_layout()
os.makedirs(args.save_dir, exist_ok=True)
plt.savefig(os.path.join(args.save_dir, "val_example.png"))
plt.show()
plt.close()
