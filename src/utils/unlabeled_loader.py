import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile


class UnlabeledTomoDataset(Dataset):
    """
    Dataset loader for unlabeled tomography images.
    Mirrors TOMODataset style but without labels.
    """

    def __init__(self, img_dir, transform=None, resized_shape=None):
        self.img_dir = img_dir
        self.transform = transform
        self.resized_shape = resized_shape
        self.imgs_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        name = self.imgs_names[idx]
        path = os.path.join(self.img_dir, name)

        img = tifffile.imread(path).astype(np.float32)

        # Same normalization used in TOMODataset
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        img = (img - p1) / (p99 - p1 + 1e-8)

        img = torch.tensor(img).float().unsqueeze(0)

        if self.resized_shape:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.resized_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.transform:
            img = self.transform(img)

        return img, name
