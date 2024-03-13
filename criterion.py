import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, ignore_index=-100) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        c = logits.size(1)
        log_preds = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (c - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

            ignore = target == self.ignore_index
            true_dist[ignore] = 0

        mask = torch.nonzero(ignore, as_tuple=False).squeeze()
        if mask.numel() > 0:
            log_preds = log_preds.index_fill(0, mask, 0.0)
            true_dist = true_dist.index_fill(0, mask, 0.0)

        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))
