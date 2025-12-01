import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..utils import TOMODataset, plot_data_transform

TRANSFORM: dict = {
    "rotation": v2.RandomRotation([-20, 20]),
    "V-flip": v2.RandomVerticalFlip(p=1.0),
    "H-flip": v2.RandomHorizontalFlip(p=1.0),
    "Affine": v2.RandomAffine(
        degrees=[-180, 180],
        translate=(0.2, 0.2),
        scale=(0.7, 1.3),
        interpolation=v2.InterpolationMode.BILINEAR,
    ),
    "stretch": v2.ElasticTransform(),
    "Gaussian-blur": v2.GaussianBlur(kernel_size=(5, 9), sigma=(1.0, 3.0)),
    "Color-jitter": v2.ColorJitter(brightness=0.5, contrast=0.5),
}
img_dir = "datas/Original Images"
label_dir = "datas/Original Masks"

# transform = v2.Compose(list(TRANSFORM.values()))
for name, transform in TRANSFORM.items():
    train_dataset = TOMODataset(img_dir, label_dir, transform=transform, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    for batch_idx, (images, labels, weights) in enumerate(train_dataloader):
        plot_data_transform(
            images, transform=name, show=False, save_dir="datas/datatransformEX"
        )
        # plot_data_transform(
        #     images,
        #     labels,
        #     weights,
        #     transform=name,
        #     show=True,
        # )
        break
