import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import decode_png
from torchvision.transforms import v2

from .weights_map_unet_paper import compute_weight_map


class UnetDataset(Dataset):
    def __init__(
        self,
        path,
        split: str,
        transform=None,
        padding: list[int] | None = None,
        resized_shape: list | None = None,
    ):
        self.img_dir = os.path.join(path, split, "imgs")
        self.label_dir = os.path.join(path, split, "labels")
        self.transform = transform
        self.padding = padding
        self.resized_shape = resized_shape

        self.imgs_names = sorted(os.listdir(self.img_dir))
        assert len(self.imgs_names) == len(sorted(os.listdir(self.label_dir))), (
            f"Mismatch length in {split} between images and labels"
        )

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
        weight_map = compute_weight_map(label.numpy())

        return image, label.squeeze(0).long(), weight_map


class TOMODataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir: Path | str | None = None,
        transform=None,
        resized_shape=None,
        split="train",
        train_ratio=0.8,
        seed=42,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.resized_shape = resized_shape
        self.split = split
        self.train_ratio = train_ratio

        self.imgs_names = sorted(os.listdir(img_dir))

        total_samples = len(self.imgs_names)
        train_size = int(self.train_ratio * total_samples)

        np.random.seed(seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        if self.split == "train":
            self.indices = indices[:train_size]
        elif self.split == "test":
            self.indices = indices[train_size:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.label_dir = label_dir
        if label_dir is not None:
            self.labels_names = sorted(os.listdir(label_dir))
            assert len(self.imgs_names) == len(self.labels_names), (
                "Mismatch in number of images and labels"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        if self.label_dir is None:
            image = self.load_image(sample_idx)
        else:
            image, label = self.load_pair(sample_idx)

        image = self.normalize(image)
        image = (
            cv2.medianBlur((image * 255).astype(np.uint8), 3) / 255.0
        )  # TODO : see if it works
        image = torch.from_numpy(image).float()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            image = image.permute(2, 0, 1)
        if self.resized_shape:
            image = v2.functional.resize(image, self.resized_shape)

        if self.label_dir is not None:
            label = torch.from_numpy(label).float()
            if label.ndim == 2:
                label = label.unsqueeze(0)
            elif label.ndim == 3:
                label = label.permute(2, 0, 1)

            if self.resized_shape:
                label = v2.functional.resize(
                    label, self.resized_shape, v2.functional.InterpolationMode.NEAREST
                )

            label = self.binarize_mask(label, 2)
            label = tv_tensors.Mask(label)

        if self.transform:
            if self.label_dir is None:
                image = self.transform(image)
                return image
            else:
                image, label = self.transform(image, label)
                weight_map = self.weights_masks(label.numpy())
                return image, label.squeeze(0).long(), weight_map

        # No transform case
        if self.label_dir is None:
            return image
        else:
            weight_map = self.weights_masks(label.numpy())
            return image, label.squeeze(0).long(), weight_map

    def load_pair(self, idx):
        img_name = self.imgs_names[idx]
        label_name = self.labels_names[idx]

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        image = tifffile.imread(img_path)
        label = tifffile.imread(label_path)

        return image, label

    def load_image(self, idx):
        img_name = self.imgs_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = tifffile.imread(img_path)
        return image

    def weights_masks(self, mask, w0=5, sigma=2):
        """mitigate class imbalance for binary set"""
        mask = mask.squeeze()
        total_pixels = mask.size
        c1_count = np.count_nonzero(mask)
        c0_count = total_pixels - c1_count

        w_c = np.zeros_like(mask, dtype=np.float32)
        w_c[mask == 0] = total_pixels / (2 * c0_count) if c0_count != 0 else 0
        w_c[mask == 1] = total_pixels / (2 * c1_count) if c1_count != 0 else 0
        w_c = np.clip(w_c, 0.1, 10.0)

        # if c1_count > 0 and c0_count > 0:
        #     # Distance from background pixels to nearest foreground object
        #     dist1 = distance_transform_edt(mask == 0)
        #     # Distance from background pixels to nearest foreground object
        #     dist2 = distance_transform_edt(mask == 1)
        #     dist = dist1 + dist2

        #     gaussian_w = w0 * np.exp((-(dist**2)) / (2 * sigma**2))
        #     weight_map = w_c + gaussian_w
        # else:
        weight_map = w_c

        # mean_weight = np.mean(weight_map)
        # if mean_weight > 0:
        #     weight_map = weight_map / mean_weight

        return torch.tensor(weight_map, dtype=torch.float32)

    def normalize(self, arr, mode: str | None = None):
        """
        Centers the dynamic range on the actual material, discard artifacts.
        """
        if mode is not None and mode == "full":
            p_lower = arr.min()
            p_upper = arr.max()
        else:
            p_lower = np.percentile(arr, 1)
            p_upper = np.percentile(arr, 99)
            arr = np.clip(arr, p_lower, p_upper)

        norm = (arr - p_lower) / (p_upper - p_lower + 1e-8)
        return norm.astype(np.float32)

    def binarize_mask(self, mask, radius):
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        else:
            mask = mask

        mask = mask.squeeze()
        threshold = threshold_otsu(mask)
        binary_mask = mask > threshold

        cleaned_mask = remove_small_objects(binary_mask, min_size=(8))

        return torch.tensor(cleaned_mask, dtype=torch.float32).unsqueeze(0)


TRANSFORM: dict = {
    "rotation": v2.RandomRotation([-20, 20]),
    "V-flip": v2.RandomVerticalFlip(p=0.5),
    "H-flip": v2.RandomHorizontalFlip(p=0.5),
    "Affine": v2.RandomAffine(
        degrees=[-180, 180],
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        interpolation=v2.InterpolationMode.BILINEAR,
    ),
    "Stretch": v2.ElasticTransform(),
    "Gaussian-blur": v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
    "Color-jitter": v2.ColorJitter(brightness=0.2, contrast=0.2),
}
