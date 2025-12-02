import os, sys, argparse, numpy as np, torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import tifffile as tiff
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
from baseline_Unet_ViT_old import U_net_ViT

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", type=str, required=True)
parser.add_argument("--unlabeled_dir", type=str, default="../datas/1h_HT")
parser.add_argument("--labeled_img_dir", type=str, default="../datas/Original Images")
parser.add_argument("--labeled_mask_dir", type=str, default="../datas/Original Masks")
parser.add_argument("--save_dir", type=str, default="models/semi_fixed")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--img_size", type=int, default=256)
args = parser.parse_args()

device = torch.device("cpu")
os.makedirs(args.save_dir, exist_ok=True)

# ==== labeled dataset ====
class LabeledDS(Dataset):
    def __init__(self, img_dir, mask_dir, size):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Look for files starting with 'image_v2_' and ending with '.tif'
        self.files = sorted([f for f in os.listdir(img_dir) if f.startswith("image_v2_") and f.lower().endswith('.tif')])
        if len(self.files) == 0:
            raise RuntimeError(f"❌ No labeled image/mask pairs found in: {img_dir} and {mask_dir}")

        self.pre = T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_name = self.files[i]
        img = tiff.imread(os.path.join(self.img_dir, img_name))
        # Replace 'image_v2_' with 'image_v2_mask_' to get the mask
        mask_name = img_name.replace("image_v2_", "image_v2_mask_")
        mask = tiff.imread(os.path.join(self.mask_dir, mask_name))

        if img.ndim == 3:
            img = img[img.shape[0]//2]
        if mask.ndim == 3:
            mask = mask[mask.shape[0]//2]

        img = img.astype(np.float32)
        if img.max() > 0:
            img /= img.max()

        mask = (mask > 0).astype(np.float32)

        im_t = self.pre((img * 255).astype(np.uint8))
        mask_t = self.pre((mask * 255).astype(np.uint8))

        return im_t.float(), mask_t.float()

# ==== unlabeled dataset ====
class UnlabeledDS(Dataset):
    def __init__(self, img_dir, size):
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.tif')])

        self.pre = T.Compose([
            T.ToPILImage(),
            T.Resize((size, size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = tiff.imread(os.path.join(self.img_dir, self.files[i]))
        if img.ndim == 3:
            img = img[img.shape[0]//2]
        img = img.astype(np.float32)
        if img.max() > 0:
            img /= img.max()
        t = self.pre((img * 255).astype(np.uint8))
        return t.mean(dim=0, keepdim=True).unsqueeze(0).float() if t.shape[0] > 1 else t.unsqueeze(0).float()

# ==== load model ====

model = U_net_ViT(
    encode_in=(1,64,128,256),
    encode_out=(64,128,256,512),
    decode_in=(1024,512,256,128),
    decode_out=(512,256,128,64),
    normalize=True
).to(device)

state = torch.load(args.pretrained, map_location="cpu")
clean_state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(clean_state, strict=True)
print("✔ Model loaded successfully")

# ==== pseudo label generation ====
print("Generating pseudo labels...")
ul_ds = UnlabeledDS(args.unlabeled_dir, args.img_size)

loader_ul = DataLoader(ul_ds, batch_size=1, pin_memory=False, num_workers=0)

pseudo_masks = []
with torch.no_grad():
    for img in tqdm(loader_ul):
    # for i, img in enumerate(loader_ul):
    #     if i >= 5:
    #         break
        img = img.squeeze(2)        # ← remove bad extra dim → [1,1,512,512]
        out = model(img)
        out = out[0] if isinstance(out, (tuple, list)) else out
        pmask = (torch.sigmoid(out) > 0.5).float()
        pseudo_masks.append(pmask.cpu())

# ==== build training set ====
# train_items = [(ul_ds[i], pseudo_masks[i], 0.6) for i in range(len(pseudo_masks))]

train_items = [(ul_ds[i], pseudo_masks[i], 0.6) for i in range(len(ul_ds))]
sup_ds = LabeledDS(args.labeled_img_dir, args.labeled_mask_dir, args.img_size)
sup_loader = DataLoader(sup_ds, batch_size=1, shuffle=True)

# ==== after defining sup_ds and sup_loader ====
val_ds = LabeledDS(args.labeled_img_dir, args.labeled_mask_dir, args.img_size)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# ==== training loop with validation ====
losses, val_losses, dices, val_dices = [], [], [], []

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0

    for (img, mask, w), (im_l, mk_l) in zip(train_items, sup_loader):
        optimizer.zero_grad()
        out = model(im_l)
        out = out[0] if isinstance(out, (tuple, list)) else out

        mask = mask.float()
        L = (F.binary_cross_entropy_with_logits(out, mask) * w) + \
            (1 - ((2*(torch.sigmoid(out)*mask).sum() + 1e-6) / (torch.sigmoid(out).sum() + mask.sum() + 1e-6)))
        L.backward()
        optimizer.step()
        running_loss += L.item()

    losses.append(running_loss)

    # ==== compute training dice metric ====
    with torch.no_grad():
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).float()
        i = (preds * mask).sum()
        u = preds.sum() + mask.sum()
        dice_score = (2 * i + 1e-6) / (u + 1e-6)
    dices.append(dice_score.item())

    # ==== validation ====
    model.eval()
    val_running_loss = 0
    val_dice_score_list = []
    with torch.no_grad():
        for val_img, val_mask in val_loader:
            val_out = model(val_img)
            val_out = val_out[0] if isinstance(val_out, (tuple, list)) else val_out
            val_mask = val_mask.float()
            val_loss = F.binary_cross_entropy_with_logits(val_out, val_mask) + \
                       (1 - ((2*(torch.sigmoid(val_out)*val_mask).sum() + 1e-6) /
                             (torch.sigmoid(val_out).sum() + val_mask.sum() + 1e-6)))
            val_running_loss += val_loss.item()

            val_probs = torch.sigmoid(val_out)
            val_preds = (val_probs > 0.5).float()
            i_val = (val_preds * val_mask).sum()
            u_val = val_preds.sum() + val_mask.sum()
            val_dice_score_list.append(((2*i_val + 1e-6)/(u_val + 1e-6)).item())

    val_losses.append(val_running_loss)
    val_dices.append(np.mean(val_dice_score_list))

    print(f"Epoch {epoch+1} | loss={running_loss:.3f} | dice≈{dices[-1]:.4f} "
          f"| val_loss={val_running_loss:.3f} | val_dice≈{val_dices[-1]:.4f}")

# ==== save outputs ====
torch.save(model.state_dict(), os.path.join(args.save_dir, "unet_vit_semi_fixed.pth"))

plt.figure(); plt.plot(losses, label="train_loss"); plt.plot(val_losses, label="val_loss")
plt.legend(); plt.savefig(os.path.join(args.save_dir,"loss_plot.png")); plt.close()

plt.figure(); plt.plot(dices, label="train_dice"); plt.plot(val_dices, label="val_dice")
plt.legend(); plt.savefig(os.path.join(args.save_dir,"dice_plot.png")); plt.close()

print("✅ Training complete, all outputs saved with validation metrics")
