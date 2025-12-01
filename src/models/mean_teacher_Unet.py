import torch
import torch.nn as nn
import copy
# from .baseline_Unet import UNet
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
from baseline_Unet_ViT_old import U_net_ViT

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
    ):
        super().__init__()

        self.ema_alpha = ema_alpha

        # Student network (the one that learns)
        # self.student = UNet(
        #     in_channels=in_channels,
        #     num_classes=num_classes,
        #     features=features,
        #     bilinear=bilinear,
        #     normalize=normalize,
        #     drop_out=drop_out,
        # )
        self.student = U_net_ViT(
            encode_in=(1,64,128,256), encode_out=(64,128,256,512),
            decode_in=(1024,512,256,128), decode_out=(512,256,128,64),
            normalize=True
        )

        # Teacher network (copy of student, not trained directly)
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

    # ============================================================
    # === CHECKPOINT LOADERS (NEW) ===============================
    # ============================================================

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


class ConsistencyLoss(nn.Module):
    """
    Consistency loss with:
      - softmax on logits,
      - temperature-based sharpening of teacher probabilities,
      - confidence mask: we only enforce consistency where teacher is confident.

    Change: try to prevent the teacher from pulling the student
    towards low-confidence or noisy predictions everywhere.
    """

    def __init__(self, temperature: float = 0.5, conf_thresh: float = 0.6):
        super().__init__()
        self.temperature = temperature
        self.conf_thresh = conf_thresh
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, student_logits, teacher_logits):
        # Student probabilities (no sharpening)
        s = torch.softmax(student_logits, dim=1)

        # Teacher probabilities, sharpened with temperature < 1
        with torch.no_grad():
            t = torch.softmax(teacher_logits / self.temperature, dim=1)
            # Confidence = max class prob
            conf, _ = t.max(dim=1, keepdim=True)  # [B,1,H,W]
            mask = (conf >= self.conf_thresh).float()

        # Per-pixel MSE between prob vectors
        # shape: [B, C, H, W] -> average over channel dim
        loss_map = self.mse(s, t).mean(dim=1, keepdim=True)  # [B,1,H,W]

        # If no confident pixels at all, fall back to unmasked mean
        if mask.sum() == 0:
            return loss_map.mean()

        loss = (loss_map * mask).sum() / mask.sum()
        return loss
