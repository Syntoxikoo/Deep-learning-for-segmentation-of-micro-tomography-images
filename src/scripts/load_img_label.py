import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ..utils import TOMODataset, plot_data_transform

img_dir = "datas/Original Images"
label_dir = "datas/Original Masks"

transform = v2.Compose(
    [
        v2.RandomRotation([-20, 20]),
        # v2.RandomVerticalFlip(p=0.5),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomAffine(
        #     degrees=180,
        #     translate=(0.1, 0.1),
        #     scale=(0.8, 1.2),
        #     interpolation=v2.InterpolationMode.BILINEAR,
        # ),
        # v2.ElasticTransform(
        #     alpha=50,
        #     sigma=5,
        # ),
        # v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
        # v2.ColorJitter(brightness=0.2, contrast=0.2),
    ]
)
train_dataset = TOMODataset(img_dir, label_dir, transform=transform, split="train")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
for batch_idx, (images, labels, weights) in enumerate(train_dataloader):
    break
