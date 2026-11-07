"""Asymmetric Loss (Ben-Baruch et al., ICCV 2021), ported from the official implementation
(https://github.com/Alibaba-MIIL/ASL).

The loss is summed over classes and averaged over images.
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()

        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric clipping -- hard-thresholds easy negatives to exactly zero loss
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            # Detach the weight from the computation graph safely
            if self.disable_torch_grad_focal_loss:
                one_sided_w = one_sided_w.detach()
                
            loss = loss * one_sided_w

        return -loss.sum(dim=1).mean()
