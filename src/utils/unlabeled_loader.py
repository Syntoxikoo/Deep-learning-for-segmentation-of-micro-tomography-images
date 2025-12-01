import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class UnlabeledTomoDataset(Dataset):
    """
    Loads unlabeled tomography images (.tif) and resizes them to a fixed size
    compatible with the U-Net training (default: 512x512).
    """

    def __init__(self, img_dir, resized_shape=(512, 512)):
        self.img_dir = img_dir
        self.resized_shape = resized_shape
        self.imgs_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        name = self.imgs_names[idx]
        path = os.path.join(self.img_dir, name)

        img = tifffile.imread(path).astype(np.float32)

        # Percentile clipping (same as supervised loader)
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-8)

        # To tensor
        img = torch.tensor(img).float().unsqueeze(0)

        # Resize to match supervised dataset
        if self.resized_shape is not None:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.resized_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img, name
