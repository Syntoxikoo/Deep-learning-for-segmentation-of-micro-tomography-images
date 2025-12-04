# import numpy as np
# import tifffile
# import matplotlib.pyplot as plt

# # ---- Unlabeled image check (10min_HT) ----
# unlabeled_path = "datas/10min_HT/10min_HT_0917.tif"
# raw_un = tifffile.imread(unlabeled_path)

# print("=== UNLABELED RAW IMAGE ===")
# print("Raw min/max:", raw_un.min(), raw_un.max())
# print("Raw percentiles (p1, p50, p99):", np.percentile(raw_un, (1, 50, 99)))

# plt.figure(figsize=(6,6))
# plt.imshow(raw_un, cmap="gray")
# plt.title("Raw Unlabeled Image")
# plt.colorbar()
# plt.show()


# # ---- Labeled image check ----
# labeled_path = "datas/Original Images/image_v2_00.tif"
# raw_l = tifffile.imread(labeled_path)

# print("\n=== LABELED RAW IMAGE ===")
# print("Raw min/max:", raw_l.min(), raw_l.max())
# print("Raw percentiles (p1, p50, p99):", np.percentile(raw_l, (1, 50, 99)))

# plt.figure(figsize=(6,6))
# plt.imshow(raw_l, cmap="gray")
# plt.title("Raw Labeled Image")
# plt.colorbar()
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
import tifffile

# --- labeled sample ---
lab = tifffile.imread("datas/Original Images/image_v2_00.tif")
plt.figure(figsize=(5,5))
plt.imshow(lab, cmap='gray')
plt.title("Labeled slice")
plt.colorbar()
plt.show()

# --- unlabeled sample ---
un = tifffile.imread("datas/10min_HT/10min_HT_0917.tif")

# center crop it EXACTLY LIKE YOUR DATALOADER
H, W = un.shape
crop_h, crop_w = 768, 768

top = (H - crop_h) // 2
left = (W - crop_w) // 2

crop_un = un[top:top+crop_h, left:left+crop_w]

plt.figure(figsize=(5,5))
plt.imshow(crop_un, cmap='gray')
plt.title("Unlabeled center-crop")
plt.colorbar()
plt.show()

