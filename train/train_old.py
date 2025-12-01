# train.py for training baseline unet with vit
# includes: deep supervision, dice over full test set, test plots, saving loss/Dice curves, eval example, no rotation-expand mismatch

# ======= TRAIN.PY =======
import os, sys, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import tifffile as tiff
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

torch.cuda.empty_cache()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../models")))
from baseline_Unet_ViT import U_net_ViT

parser = argparse.ArgumentParser(description="Train a U-Net ViT model for segmentation.")
parser.add_argument("--img_data_path", type=str, default="../datas/original/train/imgs")
parser.add_argument("--mask_data_path", type=str, default="../datas/original/train/labels")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="models/predicted_models")
args = parser.parse_args()

class MicroCTDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.tif')])
        self.mask_files = [f.replace("image_v2_", "image_v2_mask_") for f in self.img_files]

        self.preprocess = T.Resize((768,768), interpolation=T.InterpolationMode.BILINEAR)

    def __len__(self): return len(self.img_files)

    def __getitem__(self, idx):
        img = tiff.imread(os.path.join(self.img_dir, self.img_files[idx]))
        mask = tiff.imread(os.path.join(self.mask_dir, self.mask_files[idx]))
        if img.ndim==3: img = img[img.shape[0]//2]
        if mask.ndim==3: mask = mask[mask.shape[0]//2]

        img = img.astype(np.float32); img/=img.max()
        mask=(mask>0).astype(np.float32)

        img=Image.fromarray(img); mask=Image.fromarray(mask)
        img=self.preprocess(img); mask=self.preprocess(mask)
        img=torch.tensor(np.array(img)).unsqueeze(0)
        mask=torch.tensor(np.array(mask)).unsqueeze(0)

        if self.transform:
            data=torch.cat([img,mask],dim=0); data=self.transform(data)
            img,mask=data[0].unsqueeze(0), data[1].unsqueeze(0)
        return img, mask

class DiceLoss(nn.Module):
    def forward(self,p,t, smooth=1e-6):
        p=torch.sigmoid(p)
        return 1-(2*(p*t).sum()+smooth)/(p.sum()+t.sum()+smooth)

class DiceMetric:
    def __call__(self,p,t,eps=1e-6):
        p=(p.reshape(-1)>0.5).float(); t=t.reshape(-1)
        i=(p*t).sum(); u=p.sum()+t.sum()
        return ((2*i+eps)/(u+eps)).item()

dice_metric=DiceMetric()
metric=DiceMetric=DiceMetric()

# train_transform = T.Compose([
#     T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
#     T.RandomRotation(5, expand=False, fill=0),  # FIX: no expand=True mismatch at test\ň    T.RandomResizedCrop((768,768),scale=(0.9,1.0)),
#     lambda x: x + torch.randn_like(x)*0.01
# ])

train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10, fill=0, expand=False),
    T.RandomResizedCrop((768,768), scale=(0.85, 1.0)),
    T.ColorJitter(brightness=0.15, contrast=0.15),
])


dataset=MicroCTDataset(args.img_data_path, args.mask_data_path, transform=train_transform)
tbs=int(0.8*len(dataset)); vbs=len(dataset)-tbs
train_ds,val_ds=random_split(dataset,[tbs,vbs]); val_ds.dataset.transform=None
train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=1,shuffle=False)

print("Train samples:",len(train_ds),"| Val samples:",len(val_ds))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U_net_ViT(
    encode_in=(1,64,128,256), encode_out=(64,128,256,512),
    decode_in=(1024,512,256,128), decode_out=(512,256,128,64),
    normalize=True
).to(device)

optimizer=torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=5)
loss_fn=lambda p,t: nn.BCEWithLogitsLoss()(p,t)+DiceLoss()(p,t)

best_dice=0; stop_wait=15; history_losses=[]; history_val=[]; history_dice=[]; wait=0
for epoch in range(args.epochs):
    model.train(); run=0
    for img,mask in train_loader:
        img,mask=img.to(device),mask.to(device)
        optimizer.zero_grad(); p=model(img)
        mask=mask[:,:,:p.shape[2],:p.shape[3]]  # safe crop
        loss=loss_fn(p,mask); loss.backward(); optimizer.step()
        run+=loss.item()

    model.eval(); vrun=0; dices=[]
    with torch.no_grad():
        for img,mask in val_loader:
            img,mask=img.to(device),mask.to(device); p=model(img)
            mask=mask[:,:,:p.shape[2],:p.shape[3]]
            vrun+=loss_fn(p,mask).item(); dices.append(dice_metric(p,mask))
    avg_loss=run/len(train_loader); history_losses.append(avg_loss)
    avg_val=vrun/len(val_loader); history_val.append(avg_val)
    avg_dice=np.mean(dices); history_dice.append(avg_dice)
    print(f"Epoch {epoch+1}/{args.epochs} | Train: {avg_loss:.4f} | Val: {avg_val:.4f} | Dice: {avg_dice:.4f}")

    if avg_dice>best_dice: 
        best_dice=avg_dice
        wait=0
    else: 
        wait+=1;    
    if wait>=stop_wait:
        print("Early stopping epoch",epoch+1); 
        break

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.5,patience=5)
    scheduler.step(avg_dice)

scheduler.step(avg_dice); scheduler.step(avg_dice)

# Save model
mp=os.path.join(args.save_dir,"unet_vit_trained.pth"); os.makedirs(args.save_dir,exist_ok=True)
torch.save(model.state_dict(),mp); print("Model saved ✔",mp)

# Save loss plot
plt.figure(); plt.plot(history_losses,label='train'); plt.plot(history_val,label='val'); plt.legend(); plt.savefig(os.path.join(args.save_dir,"training_curves.png")); plt.close()
plt.figure(); plt.plot(history_dice,label='dice'); plt.legend(); plt.savefig(os.path.join(args.save_dir,"dice_curve.png")); plt.close()



# One eval example plot from validation
# One eval example plot from validation
img_e, mask_e = val_ds[0]                  # img_e, mask_e shapes: (1, H, W)
bimg = img_e.unsqueeze(0).to(device)      # -> (1, 1, H, W)
with torch.no_grad():
    pm = torch.sigmoid(model(bimg)).cpu() # pm shape: (1, 1, Hp, Wp)  (maybe same H/W)

# select the single image prediction (H, W)
pm_vis = pm[0, 0]  # shape: (Hp, Wp) as a tensor

# convert all to 2D numpy arrays for plotting
img_vis = img_e.squeeze(0).cpu().numpy()   # now (H, W)
mask_vis = mask_e.squeeze(0).cpu().numpy() # now (H, W)
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
plt.imshow(img_vis, cmap='gray', vmin=0.0, vmax=1.0)
plt.title("image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_vis, cmap='gray')
plt.title("mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pm_vis_np, cmap='gray', vmin=0.0, vmax=1.0)
plt.title(f"pred dice {dice_val:.4f}")
plt.axis('off')

plt.tight_layout()
os.makedirs(args.save_dir, exist_ok=True)
plt.savefig(os.path.join(args.save_dir, "val_example.png"))
plt.show()
plt.close()
