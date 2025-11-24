from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from ..utils import TOMODataset

img_dir = "datas/Original Images"
label_dir = "datas/Original Masks"

transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ]
)
train_dataset = TOMODataset(img_dir, label_dir, transform=transform, split="train")
test_dataset = TOMODataset(img_dir, label_dir, transform=None, split="test")

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

for batch_idx, (images, labels, weights) in enumerate(train_dataloader):
    if batch_idx >= 3:
        break
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(min(4, len(images))):
        # Image
        img = images[i].permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            img = img.squeeze(2)
            axes[0, i].imshow(img, cmap="gray")
        else:
            axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis("off")

        # Label
        label = labels[i].numpy()
        axes[1, i].imshow(label, cmap="gray")
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis("off")

        # Weight
        weight = weights[i].numpy()
        axes[2, i].imshow(weight)
        axes[2, i].set_title(f"weight {i+1}")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(f"src/scripts/train_batch_{batch_idx}.png")
    plt.close()
