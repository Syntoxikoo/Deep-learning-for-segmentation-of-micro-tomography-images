from torch import nn
import torch


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
        loss = self.loss_fn(logits, targets) * weights
        return loss.mean()
