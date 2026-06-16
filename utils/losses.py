import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        target = target.long().view(-1)
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)

        if self.alpha is None:
            alpha_t = 1.0
        elif isinstance(self.alpha, (float, int)):
            if logits.size(1) == 2:
                alpha_vec = logits.new_tensor([1.0 - float(self.alpha), float(self.alpha)])
                alpha_t = alpha_vec.gather(0, target)
            else:
                alpha_t = float(self.alpha)
        else:
            alpha_vec = logits.new_tensor(self.alpha)
            alpha_t = alpha_vec.gather(0, target)

        loss = -alpha_t * (1.0 - target_prob).pow(self.gamma) * target_log_prob
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
