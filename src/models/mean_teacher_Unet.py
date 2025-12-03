import copy

import torch
import torch.nn as nn

from .Unet_ViT_OLDFILE import UNetViT

class MeanTeacherUNet(nn.Module):
    """
    Wraps two UNet models:
        - student: trained via backprop
        - teacher: updated via EMA only
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        features=(64, 128, 256, 512),
        bilinear=False,
        normalize=True,
        drop_out=0.3,
        ema_alpha=0.99,
        **kwargs,
    ):
        super().__init__()

        self.ema_alpha = ema_alpha

        # Student network (the one that learns)
        self.student = UNetViT(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            normalize=normalize,
            dropout=drop_out,
            **kwargs,
        )

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
        for t_param, s_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)

    def load_student(self, ckpt_path, device="cpu"):
        """Load a saved student model checkpoint (.pth)."""
        state = torch.load(ckpt_path, map_location=device)
        self.student.load_state_dict(state)
        print(f"[MeanTeacher] Loaded student checkpoint: {ckpt_path}")

    def load_teacher(self, ckpt_path, device="cpu"):
        """Load a saved teacher model checkpoint (.pth)."""
        state = torch.load(ckpt_path, map_location=device)
        self.teacher.load_state_dict(state)
        print(f"[MeanTeacher] Loaded teacher checkpoint: {ckpt_path}")

    def load_both(self, student_ckpt, teacher_ckpt, device="cpu"):
        """Load both networks."""
        self.load_student(student_ckpt, device)
        self.load_teacher(teacher_ckpt, device)
        print("[MeanTeacher] Both student + teacher weights restored.")
