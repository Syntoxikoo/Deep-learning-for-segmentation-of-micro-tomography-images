import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation


def compute_weight_map(label, w0=10, sigma=5):
    foreground = (label < 1).astype(np.uint8)

    class_freq = np.bincount(label.flatten())
    wc = 1.0 / (class_freq[label] + 1e-6)

    # Compute borders and distances
    eroded = binary_erosion(foreground)
    border = foreground - eroded
    d1 = distance_transform_edt(1 - border)  # Distance to nearest border
    d2 = distance_transform_edt(1 - binary_dilation(border))

    # Weight map
    border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))
    w = wc + border_weight
    return torch.tensor(w, dtype=torch.float32)


def compute_weight_map(mask, w0=10, sigma=5):
    """
    Generates a weight map for the given binary mask.

    Args:
        mask (numpy array): Binary mask (H, W) where 1 is foreground, 0 is background.
        w0 (float): Weight multiplier for the boundaries. Higher = sharper edges.
        sigma (float): Controls how far the 'boundary emphasis' extends.

    Returns:
        weight_map (numpy array): The calculated weight map (H, W).
    """
    # 1. Class Balance Weights (Inverse Frequency)
    # Prevents the background from dominating the loss
    total_pixels = mask.size
    class_1_count = np.count_nonzero(mask)
    class_0_count = total_pixels - class_1_count

    # Avoid division by zero
    if class_1_count == 0:
        class_1_count = 1
    if class_0_count == 0:
        class_0_count = 1

    w_c = np.zeros_like(mask, dtype=np.float32)
    w_c[mask == 0] = total_pixels / (2 * class_0_count)  # Weight for background
    w_c[mask == 1] = total_pixels / (2 * class_1_count)  # Weight for foreground

    # 2. Distance Transform (Boundary Emphasis)
    # Calculate Euclidean distance to the nearest border
    # distance_transform_edt calculates distance to the nearest ZERO pixel.

    # Distance from background pixels to nearest foreground object
    dist1 = distance_transform_edt(mask == 0)
    # Distance from foreground pixels to nearest background (internal distance)
    dist2 = distance_transform_edt(mask == 1)

    # Combine: We want distance to the *boundary* (where values are 0)
    # This creates a map where 0 is at the edge, and value increases as you move away
    dist = dist1 + dist2

    # 3. Create the Gaussian edge weight
    # The closer to the boundary (dist near 0), the higher the weight
    w_boundary = w0 * np.exp(-(dist**2) / (2 * sigma**2))

    # 4. Final Weight Map
    weight_map = w_c + w_boundary

    return weight_map.astype(np.float32)


def get_win_coords(x, y, win_size, max_index):
    """Return window coordinates around given coordinates"""
    x_s, x_end = max(0, x - win_size), min(max_index, x + win_size)
    y_s, y_end = max(0, y - win_size), min(max_index, y + win_size)
    return x_s, x_end, y_s, y_end


def get_border_distance(
    boundary_image, pixel_labels, x, y, max_index, initial_win_size=5
):
    """
    boundary_image: Image only containing boundaries of cells
    pixel_labels: Label of each pixel identifying different cells
    """
    win_size = initial_win_size

    while True:
        x_s, x_end, y_s, y_end = get_win_coords(x, y, win_size, max_index)
        label_patch = pixel_labels[y_s:y_end, x_s:x_end]
        uni = np.unique(label_patch)
        if uni.shape[0] <= 2:
            win_size += 5
        else:
            break

    patch = boundary_image[y_s:y_end, x_s:x_end]
    patch_boundaries = patch == 1
    boundary_pixel_coords = np.where(patch_boundaries)
    boundary_pixel_label_coords = np.where(label_patch > 0)
    boundary_pixel_labels = label_patch[boundary_pixel_label_coords]

    patch_center = (win_size, win_size)
    dst_x = boundary_pixel_coords[0] - patch_center[0]
    dst_y = boundary_pixel_coords[1] - patch_center[1]
    dist = np.sqrt(dst_x**2 + dst_y**2)

    l = list(zip(dist, zip(*boundary_pixel_coords), boundary_pixel_labels))
    l.sort()

    d_1, coord_1, lab_1 = l[0]
    d_2 = None
    for d, coord, lab in l[1:]:
        if lab != lab_1:
            d_2 = d
            break

    return d_1 + (d_2 if d_2 else d_1)


def compute_weight_mapV2(label_image, w0=10, sigma=5, version=1):
    """Calculate weight map for U-Net training"""
    label_image = np.squeeze(np.array(label_image))
    max_index = label_image.shape[0] - 1

    tmp1 = label_image != 0
    struct = ndimage.generate_binary_structure(tmp1.ndim, tmp1.ndim)
    tmp2 = ndimage.binary_erosion(tmp1, struct, border_value=1)
    boundary_image = np.logical_xor(tmp1, tmp2)

    pixel_labels, _ = ndimage.label(boundary_image)

    weight_matrix = np.ones_like(
        label_image, dtype=np.float32
    )  # Start with 1.0 everywhere

    if version == 0:
        background_filt = tmp1 == False
        indices = np.where(background_filt)
    elif version == 1:
        tmp1_dil = ndimage.binary_dilation(tmp1, struct, border_value=1)
        blunt_edge_image = np.logical_xor(tmp1_dil, tmp2)
        indices = np.nonzero(blunt_edge_image)

    for y, x in zip(*indices):
        weight_matrix[y, x] = get_border_distance(
            boundary_image, pixel_labels, x, y, max_index
        )

    weight_matrix[indices] = w0 * np.exp(
        -((weight_matrix[indices]) ** 2) / (2 * (sigma**2))
    )

    return torch.tensor(weight_matrix, dtype=torch.float32)
