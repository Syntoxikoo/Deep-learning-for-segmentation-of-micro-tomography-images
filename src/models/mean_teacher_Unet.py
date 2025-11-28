import torch
import torch.nn as nn
import copy
from .baseline_Unet import UNet


class MeanTeacherUNet(nn.Module):
    """
    Wraps two UNet models:
        - student: trained via backprop
        - teacher: updated via EMA only
    """

    def __init__(self, in_channels=1, num_classes=2, features=(64,128,256,512),
                 bilinear=False, normalize=True, drop_out=0.3, ema_alpha=0.99):
        super().__init__()

        self.ema_alpha = ema_alpha

        # Student network (the one that learns)
        self.student = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            bilinear=bilinear,
            normalize=normalize,
            drop_out=drop_out,
        )

        # Teacher network (copy of student, not trained)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """
        EMA update of teacher weights:
        teacher = alpha * teacher + (1 - alpha) * student
        """
        alpha = self.ema_alpha
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)


class ConsistencyLoss(nn.Module):
    """
    MSE loss on probabilities, consistent with your WeightedCrossEntropy style.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_logits, teacher_logits):
        s = torch.softmax(student_logits, dim=1)
        t = torch.softmax(teacher_logits, dim=1)
        return self.mse(s, t)
