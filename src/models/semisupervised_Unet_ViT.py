# unet_vit_semisup.py
# Single-file module combining model (with attention, skip dropout, context) + semi-supervised training loop (Mean Teacher).
# Adjust paths, hyperparams to your environment.

import os
import math
import copy
import random
from glob import glob
from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image
import numpy as np

# ---------------------------
# Utilities: device selection
# ---------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ---------------------------
# Small helper losses
# ---------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: raw logits (B,1,H,W)
        probs = torch.sigmoid(logits)
        target = target.float()
        intersection = (probs * target).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


def bce_dice_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    dice = DiceLoss()(logits, targets)
    return bce + dice


# ---------------------------
# Model building blocks
# ---------------------------
class PrintSize(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = True
    def forward(self, x):
        if self.first:
            # keep this quiet in normal runs
            # print("Size:", x.size())
            self.first = False
        return x


class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout_rate=0.0, normalize=False):
        super().__init__()
        self.p1 = PrintSize()
        self.p2 = PrintSize()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        self.p1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout(out)
        return out


class EncodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual_channels, pool_size=2, **kwargs):
        super().__init__()
        self.conv_block = Convblock(in_ch, out_ch, **kwargs)
        self.pool = nn.MaxPool2d(pool_size)
        normalize = kwargs.get("normalize", False)
        self.resample = None
        if residual_channels != out_ch:
            self.resample = nn.Sequential(
                nn.Conv2d(out_ch, residual_channels, kernel_size=1),
                nn.BatchNorm2d(residual_channels) if normalize else nn.Identity(),
            )

    def forward(self, x):
        out = self.conv_block(x)
        pooled = self.pool(out)
        residual = self.resample(out) if self.resample is not None else out.clone()
        return pooled, residual


class DecodeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_size=2, upsampling='bilinear', **kwargs):
        super().__init__()
        self.conv_block = Convblock(in_ch, out_ch, **kwargs)
        if upsampling == "Ctranspose":
            self.up = nn.ConvTranspose2d(in_ch, in_ch, up_size, stride=up_size)
        else:
            self.up = nn.Upsample(scale_factor=up_size, mode=upsampling, align_corners=False if upsampling=='bilinear' else None)
        self.resample = None
        if up_size > 1:
            features = int(in_ch / up_size)
            self.resample = nn.Sequential(
                nn.Conv2d(in_ch, features, kernel_size=1),
                nn.BatchNorm2d(features),
            )

    def forward(self, x, residual=None):
        upsampled = self.up(x)
        if self.resample is not None:
            upsampled = self.resample(upsampled)
        if residual is not None:
            residual = F.interpolate(residual, upsampled.shape[2:], mode="bilinear", align_corners=False)
            concat = torch.cat((residual, upsampled), dim=1)
        else:
            concat = upsampled
        out = self.conv_block(concat)
        return out


# ---------------------------
# Attention Gate
# ---------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: channels of gating signal (decoder feature)
        F_l: channels of skip (encoder)
        F_int: intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder (coarse), x: encoder skip (fine)
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g1 to match x1
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # broadcast multiply



# ---------------------------
# Skip dropout / stochastic depth
# ---------------------------
class SkipDropout(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.drop = nn.Dropout2d(p)

    def forward(self, x):
        return self.drop(x)


class StochasticDepth(nn.Module):
    """Drop the entire skip tensor with probability p during training."""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, skip):
        if not self.training or self.p == 0:
            return skip
        if random.random() < self.p:
            return torch.zeros_like(skip)
        else:
            return skip


# ---------------------------
# Context block (dilated convs)
# ---------------------------
class ContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x) + self.conv2(x) + self.conv3(x)
        out = self.bn(out)
        return self.act(out)


# ---------------------------
# ViT bottleneck (MPS-safe reshape)
# ---------------------------
class ViTBottleneck(nn.Module):
    def __init__(self, channels, num_layers=2, num_heads=4, mlp_dim=None, dropout=0.1):
        super().__init__()
        d_model = channels
        dim_feedforward = 4 * d_model if mlp_dim is None else mlp_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w).permute(0, 2, 1)   # (B, N, C)
        tokens = self.encoder(tokens)
        x_out = tokens.permute(0, 2, 1).reshape(b, c, h, w)
        return x_out


class BottleneckViT(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, normalize=False, vit_num_layers=1, vit_num_heads=2, vit_mlp_dim=None, vit_dropout=0.0, filter_size=3, max_tokens=1024):
        super().__init__()
        self.max_tokens = max_tokens
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, filter_size, padding=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, filter_size, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels) if normalize else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.context = ContextBlock(bottleneck_channels)   # small context module
        self.vit = ViTBottleneck(channels=bottleneck_channels, num_layers=vit_num_layers, num_heads=vit_num_heads, mlp_dim=vit_mlp_dim, dropout=vit_dropout)

    def forward(self, x):
        b, c, h0, w0 = x.shape
        x_down = x
        # adaptively downsample
        while (x_down.shape[2] * x_down.shape[3] > self.max_tokens) and (x_down.shape[2] >= 4 and x_down.shape[3] >= 4):
            x_down = F.max_pool2d(x_down, kernel_size=2, stride=2)
        # convs
        x_down = self.conv1(x_down); x_down = self.bn1(x_down); x_down = self.act(x_down)
        x_down = self.conv2(x_down); x_down = self.bn2(x_down); x_down = self.act(x_down)
        # context
        x_down = self.context(x_down)
        # vit
        x_down = self.vit(x_down)
        # upsample
        x_up = F.interpolate(x_down, size=(h0, w0), mode='bilinear', align_corners=False)
        return x_up


# ---------------------------
# Full U-net with ViT bottleneck + attention + skip dropout
# ---------------------------
class U_net_ViT(nn.Module):
    def __init__(self, encode_in=(1,), encode_out=(64,), decode_in=(128,), decode_out=(64,), filter_size=3,
                 vit_num_layers=1, vit_num_heads=2, vit_mlp_dim=None, vit_dropout=0.0,
                 skip_dropout_prob=0.1, skip_stochastic_prob=0.0, normalize=True, bottleneck_channels=None, max_tokens=1024):
        super().__init__()
        assert len(encode_in) == len(decode_in)
        self.N_layers = len(encode_in)
        self.encode = nn.ModuleList()
        for ii in range(self.N_layers):
            self.encode.append(EncodeBlock(in_ch=encode_in[ii], out_ch=encode_out[ii],
                                           residual_channels=int(decode_in[self.N_layers - 1 - ii] / 2),
                                           normalize=normalize))
        last_encode = encode_out[-1]
        if bottleneck_channels is None:
            bottleneck_channels = last_encode * 2
        self.bottleneck = BottleneckViT(in_channels=last_encode, bottleneck_channels=bottleneck_channels,
                                        normalize=normalize, vit_num_layers=vit_num_layers, vit_num_heads=vit_num_heads,
                                        vit_mlp_dim=vit_mlp_dim, vit_dropout=vit_dropout, filter_size=filter_size, max_tokens=max_tokens)
        self.decode = nn.ModuleList()
        for ii in range(self.N_layers):
            self.decode.append(DecodeBlock(in_ch=decode_in[ii], out_ch=decode_out[ii], normalize=normalize))
        # attention gates for each skip connection (mirror sizes)
        self.attentions = nn.ModuleList()
        for ii in range(self.N_layers):
            # decoder feature channels = decode_in[ii] (approx), skip channels = encode_out[self.N_layers-1-ii]
            F_g = decode_in[ii]
            F_l = encode_out[self.N_layers - 1 - ii]
            F_int = max(8, F_g // 2)
            self.attentions.append(AttentionGate(F_g=F_g, F_l=F_l, F_int=F_int))
        self.skip_dropout = SkipDropout(p=skip_dropout_prob)
        self.skip_stochastic = StochasticDepth(p=skip_stochastic_prob)
        self.segment_conv = nn.Conv2d(decode_out[-1], 1, kernel_size=1)

    def forward(self, x):
        residuals = []
        for ii in range(self.N_layers):
            x, residual = self.encode[ii](x)
            residuals.append(residual)
        x = self.bottleneck(x)
        residuals = residuals[::-1]
        for ii in range(self.N_layers):
            skip = residuals[ii]
            # apply skip dropout / stochastic depth
            skip = self.skip_dropout(skip)
            skip = self.skip_stochastic(skip)
            # attention gate: decoder feature 'x' is gating, skip is encoder feature
            att = self.attentions[ii]
            skip_att = att(x, skip)
            x = self.decode[ii](x, skip_att)
        x = self.segment_conv(x)
        return x


# ---------------------------
# Data loading (labeled + unlabeled .tif)
# ---------------------------
class LabeledTifDataset(Dataset):
    def __init__(self, images: List[str], masks: List[str], transform=None):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx]).convert("L")
        img = img.convert("L")  # single channel
        if self.transform:
            res = self.transform(image=np.array(img), mask=np.array(mask))
            img = res['image']
            mask = res['mask']
        else:
            img = T.ToTensor()(img)
            mask = (T.ToTensor()(mask) > 0.5).float()
        return img, mask


class UnlabeledTifDataset(Dataset):
    def __init__(self, images: List[str], weak_transform=None, strong_transform=None):
        self.images = images
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("L")
        arr = np.array(img)
        # weak augmented version (for teacher)
        if self.weak_transform:
            w = self.weak_transform(image=arr)['image']
        else:
            w = T.ToTensor()(img)
        # strong augmented version (for student)
        if self.strong_transform:
            s = self.strong_transform(image=arr)['image']
        else:
            s = T.ToTensor()(img)
        return w, s


# ---------------------------
# Simple augmentations using torchvision + albumentations style wrapper
# (I keep transforms minimal — replace with albumentations for more power)
# ---------------------------
def make_transforms(image_size=384):
    # weak color / spatial transforms
    weak = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
    ])
    # strong: include blur, jitter, maybe elastic (if desired)
    strong = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
        T.ToTensor(),
    ])
    # for labeled pairs: same transform on both image and mask — use a deterministic wrapper in real runs
    def labeled_transform(image, mask):
        # image, mask are numpy arrays — convert to PIL then apply identical torchvision transforms by seeding or using albumentations.
        pil_img = Image.fromarray(image).convert("L")
        pil_mask = Image.fromarray(mask).convert("L")
        # we will do simple deterministic ops to keep masks aligned:
        transform = T.Compose([T.Resize((image_size, image_size))])
        img_t = transform(pil_img)
        mask_t = transform(pil_mask)
        img_t = T.ToTensor()(img_t)
        mask_t = (T.ToTensor()(mask_t) > 0.5).float()
        return {'image': img_t, 'mask': mask_t}

    # wrappers returning callables similar to albumentations
    weak_wrapper = lambda image: {'image': weak(Image.fromarray(image).convert("L"))}
    strong_wrapper = lambda image: {'image': strong(Image.fromarray(image).convert("L"))}
    return labeled_transform, weak_wrapper, strong_wrapper


# ---------------------------
# Semi-supervised training loop (Mean Teacher)
# ---------------------------
def update_ema(student, teacher, alpha=0.99):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = alpha * t_param.data + (1.0 - alpha) * s_param.data


def train_semi(
    student: nn.Module,
    teacher: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer,
    num_epochs=2,
    unsup_weight_max=1.0,
    device=torch.device('cpu')
):
    student.to(device)
    teacher.to(device)
    teacher.eval()
    global_step = 0
    for epoch in range(num_epochs):
        student.train()
        total_sup_loss = 0.0
        total_cons_loss = 0.0
        for (x_l, y_l), (w_u, s_u) in zip(labeled_loader, unlabeled_loader):
            # ensure tensors on device
            x_l = x_l.to(device); y_l = y_l.to(device)
            w_u = w_u.to(device); s_u = s_u.to(device)

            optimizer.zero_grad()
            # supervised forward
            logits = student(x_l)
            sup_loss = bce_dice_loss(logits, y_l)

            # teacher produces pseudo-targets from weakly augmented unlabeled images
            with torch.no_grad():
                teacher_logits = teacher(w_u)
                teacher_probs = torch.sigmoid(teacher_logits)

            # student predicts on strongly augmented unlabeled images
            student_logits_unl = student(s_u)
            student_probs_unl = torch.sigmoid(student_logits_unl)

            # consistency loss (MSE)
            cons_loss = F.mse_loss(student_probs_unl, teacher_probs)

            # confidence mask (optionally mask teacher low-confidence)
            # mask = (teacher_probs > 0.7).float()
            # cons_loss = ( (student_probs_unl - teacher_probs)**2 * mask ).mean()

            # ramp up unsupervised weight
            p = epoch / float(num_epochs)
            unsup_weight = unsup_weight_max * float(math.exp(-5 * (1 - p)**2))  # sigmoidal ramp up

            loss = sup_loss + unsup_weight * cons_loss
            loss.backward()
            optimizer.step()

            # EMA update teacher
            update_ema(student, teacher, alpha=0.99)

            total_sup_loss += sup_loss.item()
            total_cons_loss += cons_loss.item()
            global_step += 1

        print(f"Epoch {epoch+1}/{num_epochs} | Sup Loss: {total_sup_loss/len(labeled_loader):.4f} | Cons Loss: {total_cons_loss/len(unlabeled_loader):.4f}")

    return student, teacher


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # =========== paths ===========
    labeled_images = sorted(glob("datas/Original Images/*.tif"))   # adapt
    labeled_masks = sorted(glob("datas/Original Masks/*.tif"))     # adapt
    unlabeled_path = "datas/1h_HT"  # user-provided path in your message
    unlabeled_images = sorted(glob(os.path.join(unlabeled_path, "*.tif")))

    # quick sanity
    print("Labeled:", len(labeled_images), "Unlabeled:", len(unlabeled_images))

    # =========== transforms ===========
    labeled_transform, weak_transform, strong_transform = make_transforms(image_size=384)

    labeled_ds = LabeledTifDataset(labeled_images, labeled_masks, transform=labeled_transform)
    unlabeled_ds = UnlabeledTifDataset(unlabeled_images, weak_transform=weak_transform, strong_transform=strong_transform)

    labeled_loader = DataLoader(labeled_ds, batch_size=1, shuffle=True, num_workers=0)   # MPS: num_workers=0
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=1, shuffle=True, num_workers=0)

    # =========== model ===========
    # A small configuration to avoid MPS OOM — lower channels / tokens
    encode_in = (1, 64)      # two encode layers example; adapt to your prior network shape
    encode_out = (64, 128)
    decode_in = (256, 128)
    decode_out = (128, 64)

    student = U_net_ViT(encode_in=encode_in, encode_out=encode_out, decode_in=decode_in, decode_out=decode_out,
                       vit_num_layers=1, vit_num_heads=2, vit_mlp_dim=None, vit_dropout=0.0,
                       skip_dropout_prob=0.1, skip_stochastic_prob=0.0, normalize=True, max_tokens=1024)
    teacher = copy.deepcopy(student)
    # teacher param requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    student = student.to(device)
    teacher = teacher.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    # =========== train semi-supervised ===========
    train_semi(student, teacher, labeled_loader, unlabeled_loader, optimizer, num_epochs=25, unsup_weight_max=1.0, device=device)

    # Save student
    torch.save(student.state_dict(), "./unet_vit_semisup_student.pth")
    torch.save(teacher.state_dict(), "./unet_vit_semisup_teacher.pth")
    print("Saved models.")
