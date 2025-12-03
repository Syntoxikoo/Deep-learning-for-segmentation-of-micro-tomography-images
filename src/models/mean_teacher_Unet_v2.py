import torch
import torch.nn as nn
import copy


class MeanTeacherUNetV2(nn.Module):
    """
    Minimal Mean Teacher wrapper around your UNet-ViT model.
    """

    def __init__(self, student_model: nn.Module, ema_alpha: float = 0.99):
        super().__init__()

        self.student = student_model
        self.teacher = copy.deepcopy(student_model)
        self.ema_alpha = ema_alpha

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update: teacher = α * teacher + (1-α) * student."""
        alpha = self.ema_alpha
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(alpha).add_(s_p.data, alpha=1.0 - alpha)

    def load_student(self, path, device="cpu"):
        state = torch.load(path, map_location=device)
        self.student.load_state_dict(state)
        print(f"[MeanTeacherV2] loaded student from {path}")

    def load_teacher(self, path, device="cpu"):
        state = torch.load(path, map_location=device)
        self.teacher.load_state_dict(state)
        print(f"[MeanTeacherV2] loaded teacher from {path}")
