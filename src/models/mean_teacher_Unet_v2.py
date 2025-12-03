import copy
import torch
import torch.nn as nn

from .Unet_ViT import U_net_ViT


class MeanTeacherUNetV2(nn.Module):
    """
    Mean Teacher wrapper for the NEW U_net_ViT (1-channel output + deep supervision).
    """

    def __init__(
        self,
        ema_alpha=0.99,
        **kwargs
    ):
        super().__init__()

        self.ema_alpha = ema_alpha

        # ---- student ----
        self.student = U_net_ViT(**kwargs)

        # ---- teacher ----
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        alpha = self.ema_alpha
        for t, s in zip(self.teacher.parameters(), self.student.parameters()):
            t.data.mul_(alpha).add_(s.data, alpha=1 - alpha)

    # Optional loading helpers
    def load_student(self, ckpt_path, device="cpu"):
        state = torch.load(ckpt_path, map_location=device)
        self.student.load_state_dict(state)
        print(f"Loaded student from {ckpt_path}")

    def load_teacher(self, ckpt_path, device="cpu"):
        state = torch.load(ckpt_path, map_location=device)
        self.teacher.load_state_dict(state)
        print(f"Loaded teacher from {ckpt_path}")
