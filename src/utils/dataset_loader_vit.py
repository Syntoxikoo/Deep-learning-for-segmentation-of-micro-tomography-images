# dataset_loader.py load dataset for unet and custom loaders
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
import os
from torchvision.io import decode_png
from torchvision.transforms import v2
# from .weights_map_unet_paper import compute_weight_map
import tifffile
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk
from typing import Optional, List
import random


class UnetDatasetViT(Dataset):

    def __init__(
        self,
        path,
        split: str,
        transform=None,
        padding: Optional[int] = None,
        resized_shape: Optional[List[int]] = None
    ):
        self.img_dir = os.path.join(path, split, "imgs")
        self.label_dir = os.path.join(path, split, "labels")
        self.transform = transform
        self.padding = padding
        self.resized_shape = resized_shape

        self.imgs_names = sorted(os.listdir(self.img_dir))
        assert len(self.imgs_names) == len(
            sorted(os.listdir(self.label_dir))
        ), f"Mismatch length in {split} between images and labels"

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_name = self.imgs_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        img_png = open(img_path, "rb").read()
        label_png = open(label_path, "rb").read()
        img_tensor = torch.frombuffer(img_png, dtype=torch.uint8).clone()
        label_tensor = torch.frombuffer(label_png, dtype=torch.uint8).clone()

        image = decode_png(img_tensor)
        image = v2.functional.to_dtype(image, torch.float32, scale=True)
        label = decode_png(label_tensor)

        if self.resized_shape:
            image = v2.functional.resize(image, self.resized_shape)
            label = v2.functional.resize(label, self.resized_shape)

        if self.padding:
            image = v2.functional.pad(image, self.padding, fill=0)
            label = v2.functional.pad(label, self.padding, fill=0)

        label = tv_tensors.Mask(label)

        if self.transform:
            image, label = self.transform(image, label)

        if self.padding and self.resized_shape:
            label = v2.functional.center_crop(label, self.resized_shape)

        label = (label >= 125).long()  # binarize
        
        # return image, label.squeeze(0).long(), weight_map
        return image, label.squeeze(0).long()



class TOMODatasetViT(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        transform=None,
        resized_shape=None,
        split="train",
        train_ratio=0.8,
        seed=42,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.resized_shape = resized_shape
        self.split = split
        self.train_ratio = train_ratio

        self.imgs_names = sorted(os.listdir(img_dir))
        self.labels_names = sorted(os.listdir(label_dir))
        assert len(self.imgs_names) == len(
            self.labels_names
        ), "Mismatch in number of images and labels"

        total_samples = len(self.imgs_names)
        train_size = int(self.train_ratio * total_samples)

        np.random.seed(seed)
        indices = np.random.permutation(total_samples)

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "test":
            self.indices = indices[train_size:]
        elif split is None:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 1) Get the sample index
        sample_idx = self.indices[idx]

        # 2) Load image + label pair (NumPy arrays)
        image, label = self.load_pair(sample_idx)  # image: (H,W), label: (H,W), both numpy

        # 3) If volumetric, select a random slice
        if image.ndim == 3:  # (D,H,W)
            slice_idx = random.randint(0, image.shape[0]-1)
            image = image[slice_idx]
            label = label[slice_idx]

        # 4) Normalize the image (still numpy!)
        image = self.normalize(image)

        # 5) Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1,H,W)
        label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)  # (1,H,W)

        # 6) Resize if needed
        if self.resized_shape is not None:
            image = v2.functional.resize(
                image, self.resized_shape, interpolation=v2.functional.InterpolationMode.BILINEAR
            )
            label = v2.functional.resize(
                label, self.resized_shape, interpolation=v2.functional.InterpolationMode.NEAREST
            )

        # 7) Albumentations transform (optional)
        if self.transform:
            img_np = image.squeeze(0).numpy()[:, :, None]   # (H,W,1)
            mask_np = label.squeeze(0).numpy()[:, :, None]
            aug = self.transform(image=img_np, mask=mask_np)
            image = torch.from_numpy(aug["image"]).permute(2,0,1).float()
            label = torch.from_numpy(aug["mask"]).permute(2,0,1).float()

        # 8) Final binarization for mask
        label = (label > 0.5).float()  # (1,H,W)

        return image, label



    # def __getitem__(self, idx):
    #     sample_idx = self.indices[idx]
    #     image, label = self.load_pair(sample_idx)

    #     # ---------------------------------------------------------
    #     # 1) ALWAYS SLICE 3D → 2D BEFORE ANYTHING ELSE
    #     # ---------------------------------------------------------
    #     if image.ndim == 3:     # (D, H, W)
    #         slice_idx = random.randint(0, image.shape[0] - 1)
    #         image = image[slice_idx]
    #         label = label[slice_idx]

    #     # Now BOTH are guaranteed (H, W)
    #     assert image.ndim == 2 and label.ndim == 2, \
    #         f"Still not 2D: {image.shape}, {label.shape}"

    #     # ---------------------------------------------------------
    #     # 2) Normalize BEFORE torch conversion
    #     # ---------------------------------------------------------
    #     image = self.normalize(image)
    #     label = label.astype(np.float32)

    #     # ---------------------------------------------------------
    #     # 3) Convert to torch (1, H, W)
    #     # ---------------------------------------------------------
    #     image = torch.from_numpy(image).float().unsqueeze(0)
    #     label = torch.from_numpy(label).float().unsqueeze(0)

    #     # ---------------------------------------------------------
    #     # 4) Resize (if needed)
    #     # ---------------------------------------------------------
    #     if self.resized_shape is not None:
    #         image = v2.functional.resize(
    #             image, self.resized_shape,
    #             interpolation=v2.functional.InterpolationMode.BILINEAR
    #         )
    #         label = v2.functional.resize(
    #             label, self.resized_shape,
    #             interpolation=v2.functional.InterpolationMode.NEAREST
    #         )

    #     # ---------------------------------------------------------
    #     # 5) Albumentations expects numpy channels-last
    #     # ---------------------------------------------------------
    #     if self.transform:
    #         img_np = image.squeeze(0).numpy()       # (H, W)
    #         mask_np = label.squeeze(0).numpy()

    #         img_np = img_np[:, :, None]             # → (H, W, 1)
    #         mask_np = mask_np[:, :, None]

    #         aug = self.transform(image=img_np, mask=mask_np)

    #         image = torch.from_numpy(aug["image"]).permute(2, 0, 1).float()
    #         label = torch.from_numpy(aug["mask"]).permute(2, 0, 1).float()

    #     # ---------------------------------------------------------
    #     # 6) Mask binarization using Otsu on 2D only
    #     # ---------------------------------------------------------
    #     label = self.binarize_mask(label.squeeze(0))   # → (1, H, W)

    #     print("unique mask:", torch.unique(label))

    #     return image, label.long()

    # ======================================================================
    # Utilities
    # ======================================================================

    def load_pair(self, idx):
        """Load one volumetric image + label pair"""
        img_path = os.path.join(self.img_dir, self.imgs_names[idx])
        label_path = os.path.join(self.label_dir, self.labels_names[idx])

        image = tifffile.imread(img_path)
        label = tifffile.imread(label_path)

        label = 1 - label 

        label = (label > 0).astype(np.uint8)

        return image, label

    def normalize(self, arr):
        """Percentile normalization (1–99%)"""
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)
        return ((arr - p1) / (p99 - p1)).astype(np.float32)

    # def binarize_mask(self, mask):
    #     """Otsu + morphological cleaning"""

    #     if isinstance(mask, torch.Tensor):
    #         mask = mask.cpu().numpy()

    #     if mask.ndim != 2:
    #         raise ValueError(f"Mask must be 2D in binarize_mask, got {mask.shape}")

    #     thresh = threshold_otsu(mask)
    #     binary = mask > thresh
    #     cleaned = binary_opening(binary, footprint=disk(1))

    #     return torch.tensor(cleaned, dtype=torch.float32).unsqueeze(0)
    def binarize_mask(self, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.float()
        return (mask > 0.5).float().unsqueeze(0)

    # def binarize_mask(self, mask):
    #     """Simple threshold + morphological cleaning. Accepts torch or numpy 2D mask."""
    #     # convert to numpy 2D array
    #     if isinstance(mask, torch.Tensor):
    #         mask_np = mask.detach().cpu().numpy()
    #     else:
    #         mask_np = np.array(mask)

    #     # handle possible shapes: (1,H,W), (H,W), (H,W,1)
    #     if mask_np.ndim == 3 and mask_np.shape[0] == 1:   # (1,H,W)
    #         mask_np = mask_np[0]
    #     if mask_np.ndim == 3 and mask_np.shape[2] == 1:   # (H,W,1)
    #         mask_np = mask_np[:, :, 0]

    #     if mask_np.ndim != 2:
    #         raise ValueError(f"Mask must be 2D in binarize_mask, got {mask_np.shape}")

    #     # if mask is already binary (0/1), use simple threshold 0.5
    #     # otherwise, scale to 0..1 if necessary and threshold at 0.5
    #     if mask_np.dtype == np.bool_ or set(np.unique(mask_np)).issubset({0, 1}):
    #         binary = mask_np > 0
    #     else:
    #         # normalize to 0..1 if big range, then threshold
    #         mn, mx = mask_np.min(), mask_np.max()
    #         if mx > 1.0:
    #             mask_norm = (mask_np - mn) / (mx - mn + 1e-8)
    #         else:
    #             mask_norm = mask_np
    #         binary = mask_norm > 0.5

    #     # small morphological cleaning
    #     cleaned = binary_opening(binary, footprint=disk(1))

    #     return torch.tensor(cleaned, dtype=torch.float32).unsqueeze(0)

