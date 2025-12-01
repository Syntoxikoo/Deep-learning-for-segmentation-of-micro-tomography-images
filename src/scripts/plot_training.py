import numpy as np

from src.utils.plot_func import plot_losses_curves

# Load data from CSV - e.g. :
data = np.genfromtxt("checkpoints/run_01-14h09/metrics.csv", delimiter=",", names=True)

train_loss = data["train_loss"]
val_loss = data["val_loss"]

# Plot training and validation loss curves
plot_losses_curves(train_loss, val_loss, "checkpoints", show=True)
