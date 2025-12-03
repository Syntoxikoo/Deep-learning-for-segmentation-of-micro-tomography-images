import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, List, Union

def plot_losses_curves(train_losses, val_losses, save_dir, show=False):
    plt.figure(figsize=(10, 6))
    kwargs = {
        "marker": "o",
        "markeredgecolor": "black",
        "alpha": 0.9,
        "markeredgewidth": 0.4,
    }
    plt.plot(train_losses, label="Training Loss", **kwargs)
    plt.plot(val_losses, label="Validation Loss", **kwargs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_curves.svg"),
        dpi=600,
        format="svg",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()


def plot_prediction(image, label, pred, save_dir, epoch=None, show=False):
    # Ensure format
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    kwargs = {
        "fontsize": 12,
        "fontweight": "bold",
    }
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image", **kwargs)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth", **kwargs)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
    plt.title("Prediction", **kwargs)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"prediction_plot{epoch}.svg"),
        dpi=600,
        format="svg",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()

def plot_prediction_v2(image, label, pred_logits, save_dir, epoch=None, show=False):
    # Convert tensors
    image = image.squeeze().cpu().numpy()
    label = label.squeeze().cpu().numpy()

    # Always convert logits → binary mask
    if isinstance(pred_logits, torch.Tensor):
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze().cpu().numpy()
    else:
        pred_mask = pred_logits  # Already a numpy array or mask

    plt.figure(figsize=(15, 5))
    kwargs = {"fontsize": 12, "fontweight": "bold"}

    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image", **kwargs)
    plt.axis("off")

    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth", **kwargs)
    plt.axis("off")

    # Prediction (binary)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Prediction", **kwargs)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"prediction_plot_{epoch}.svg"),
        dpi=600,
        format="svg",
        bbox_inches="tight",
    )
    if show:
        plt.show()
    plt.close()


def plot_data_transform(
    image,
    label=None,
    weight=None,
    transform: Optional[str] = None,
    save_dir=None,
    show=False,
):
    """
    Plot data transformations for a batch of images, optionally including labels and weights.

    Parameters:
    - images: Tensor of shape (batch_size, C, H, W)
    - labels: Tensor of shape (batch_size, H, W) or None
    - weights: Tensor of shape (batch_size, H, W) or None
    - transform: Transform name
    - save_dir: Directory to save the plot
    - show: Whether to display the plot
    """
    cols = 1
    image = image.squeeze().cpu().numpy()
    datas = [image]

    titles = ["Image"]
    if label is not None:
        cols += 1
        label = label.squeeze().cpu().numpy()
        datas.append(label)
        titles.append("Label")
    if weight is not None:
        cols += 1
        weight = weight.squeeze().cpu().numpy()
        datas.append(weight)
        titles.append("Weight")
    figwidth = 4 * cols
    fig, axes = plt.subplots(1, cols, figsize=(figwidth, 4))

    if cols == 1:
        axes = [axes]

    for i in range(cols):
        axes[i].imshow(datas[i], cmap="gray", vmin=0, vmax=1)
        if cols > 1:
            axes[i].set_title(titles[i], fontsize=14, fontweight="bold")
        axes[i].axis("off")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"plot_data_{transform}.svg"),
            dpi=600,
            format="svg",
            bbox_inches="tight",
        )
    if show:
        plt.show()
    plt.close()


def visualize_student_vs_teacher(model, unlabeled_loader, device, save_dir):
    """
    Visualizes how the Student and Teacher differ on UNLABELED data.
    """
    model.eval()
    model.student.eval()
    model.teacher.eval()

    iterator = iter(unlabeled_loader)

    for i in range(3):
        try:
            images_u = next(iterator)
        except StopIteration:
            break

        images_u = images_u.to(device)

        with torch.no_grad():
            stu_logits, _ = model.student(images_u)
            tea_logits, _ = model.teacher(images_u)

            # Get predictions (argmax)
            stu_pred = torch.argmax(stu_logits, dim=1)[0].cpu().numpy()
            tea_pred = torch.argmax(tea_logits, dim=1)[0].cpu().numpy()
            img_np = images_u[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Unlabeled Input")

        axes[1].imshow(stu_pred, cmap="gray")
        axes[1].set_title("Student Prediction")

        axes[2].imshow(tea_pred, cmap="gray")
        axes[2].set_title("Teacher Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"vis_unlabeled_example_{i}.svg"),
            dpi=600,
            format="svg",
            bbox_inches="tight",
        )
        plt.close()
