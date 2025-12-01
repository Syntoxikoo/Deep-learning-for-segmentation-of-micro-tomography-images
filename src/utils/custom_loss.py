import torch
from torch import nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits, targets, weights):
        probs = self.softmax(logits)
        log_probs = torch.log(probs + 1e-10)
        loss = -weights.unsqueeze(1) * log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        return loss.mean()


class WeightedCrossEntropyLossV2(nn.Module):
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

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)

        input_obj = inputs[:, 1, :, :]
        target_obj = (targets == 1).float()

        intersection = (input_obj * target_obj).sum()
        dice = (2.0 * intersection) / (input_obj.sum() + target_obj.sum() + 1e-8)
        return 1 - dice
