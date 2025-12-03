import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """
    Consistency loss with:
      - softmax on logits,
      - temperature-based sharpening of teacher probabilities,
      - confidence mask: we only enforce consistency where
    """

    def __init__(self, temperature: float = 0.5, conf_thresh: float = 0.6):
        super().__init__()
        self.temperature = temperature
        self.conf_thresh = conf_thresh
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, student_logits, teacher_logits):
        s = torch.softmax(student_logits, dim=1)

        with torch.no_grad():
            t = torch.softmax(teacher_logits / self.temperature, dim=1)

            # Get probability of the predicted class
            max_prob, _ = t.max(dim=1, keepdim=True)
            mask = (max_prob >= self.conf_thresh).float()

        # 3. Calculate MSE
        loss_map = self.mse(s, t).mean(dim=1, keepdim=True)

        # 4. Masked Average
        mask_sum = mask.sum()

        if mask_sum < 1e-6:
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # Only average over pixels that crossed the threshold
        loss = (loss_map * mask).sum() / mask_sum
        return loss


class WeightCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, weights):
        loss = self.loss_fn(logits, targets)
        w_loss = loss * weights.float()
        f_loss = w_loss.sum() / (weights.sum() + 1e-8)
        return f_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights: None = None):
        inputs = torch.softmax(inputs, dim=1)

        input_obj = inputs[:, 1, :, :]
        target_obj = (targets == 1).float()

        intersection = (input_obj * target_obj).sum()
        dice = (2.0 * intersection) / (input_obj.sum() + target_obj.sum() + 1e-8)
        return 1 - dice


class DiceMetric:
    """Dice coefficient metric for multi-class segmentation (binary foreground).

    Computes the Dice coefficient for class 1 (foreground) using argmax predictions
    and binary targets. This is the standard metric for semantic segmentation tasks.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, predictions, targets):
        # Get hard class predictions using argmax
        pred_classes = torch.argmax(predictions, dim=1)  # (B, H, W)

        intersection = ((pred_classes == 1) & (targets == 1)).sum().float()
        union = (pred_classes == 1).sum().float() + (targets == 1).sum().float()

        dice = (2.0 * intersection) / (union + self.eps)

        return dice.item()
