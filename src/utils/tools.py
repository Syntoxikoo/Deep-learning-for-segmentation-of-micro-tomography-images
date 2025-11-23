import logging
import os
import sys
import tifffile

import torch
from torch import nn
from torchvision.transforms.v2 import Compose
import matplotlib.pyplot as plt


class PairTransform:
    def __init__(self, transform_pipeline: Compose):
        self.transform_pipeline = transform_pipeline

    def __call__(self, image, label):
        if label.ndim == 2:
            label = label.unsqueeze(0)

        image, label = self.transform_pipeline(image, label)

        return image, label.squeeze(0)


def get_device():
    """get device for os and N dataloader workers

    Returns:
        tuple: (device,dl_workers)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
        dl_workers = 0
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
        dl_workers = 4
    else:
        device = torch.device("cpu")
        print("Using CPU")
        dl_workers = 4
    return device, dl_workers


def init_weights(m):
    """Initialize Conv and BatchNorm weights sensibly."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def setup_logger(save_dir):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(os.path.join(save_dir, "training_log.txt"))
    fh.setLevel(logging.INFO)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def visualize_encoder_features(
    model,
    image,
    save_path,
    device="cpu",
    max_channels=8,
):
    """
    Passes an image through the UNet encoder and visualizes the feature maps.

    Args:
        model: The trained UNet model.
        image: Input image tensor (1, C, H, W).
        device: mps, cuda,cpu
        max_channels: Number of feature map channels to visualize per layer.
        save_path: saves the figures to this path.
    """
    model.eval()
    image = image.to(device)

    feature_maps = []
    layer_names = []

    with torch.no_grad():
        x = model.inc(image)
        feature_maps.append(x)
        layer_names.append("Encoder Layer 1 (Inc)")

        curr_x = x
        for i, down_layer in enumerate(model.downs):
            conv_out, pooled = down_layer(curr_x)
            feature_maps.append(conv_out)
            layer_names.append(f"Encoder Layer {i+2} (Down)")
            curr_x = pooled

    figs = []
    for layer_idx, (fmap, name) in enumerate(zip(feature_maps, layer_names)):
        fmap = fmap.squeeze(0).cpu()
        num_channels = fmap.shape[0]
        channels_to_show = min(max_channels, num_channels)

        cols = 4
        rows = (channels_to_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        fig.suptitle(f"{name} (Top {channels_to_show} channels)", fontsize=16)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for channel_idx in range(channels_to_show):
            row = channel_idx // cols
            col = channel_idx % cols
            ax = axes[row][col]

            activation = fmap[channel_idx]
            activation = (activation - activation.min()) / (
                activation.max() - activation.min() + 1e-8
            )
            ax.imshow(activation.numpy(), cmap="viridis")
            ax.axis("off")
            ax.set_title(f"Ch {channel_idx}", fontsize=8)

        # Turn off unused subplots
        for channel_idx in range(channels_to_show, rows * cols):
            row = channel_idx // cols
            col = channel_idx % cols
            axes[row][col].axis("off")

        figs.append(fig)

    for i, fig in enumerate(figs):
        fig.savefig(f"{save_path}_layer_{i}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
    return save_path


def load_tiff_image(path):
    img = tifffile.imread(path)
    if img.ndim == 3:  # takes middle slicee in case of 3D.
        img = img[img.shape[0] // 2]
    return img
