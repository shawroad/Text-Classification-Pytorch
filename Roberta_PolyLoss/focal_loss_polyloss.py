import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = torch.zeros(num_classes)
        self.alpha[0] += alpha
        self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        self.alpha = self.alpha.to(logits.device)
        logits_logsoft = F.log_softmax(logits, dim=1)
        logits_softmax = torch.exp(logits_logsoft)
        logits_softmax = logits_softmax.gather(1, labels.view(-1, 1))
        logits_logsoft = logits_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - logits_softmax), self.gamma), logits_logsoft)
        loss = torch.mul(self.alpha, loss.t())[0, :]
        return loss


def poly1_focal_loss_torch(logits, labels, alpha=0.25, gamma=2, num_classes=3, epsilon=1.0):
    focal_loss_func = FocalLoss(alpha, gamma, num_classes)
    focal_loss = focal_loss_func(logits, labels)

    p = torch.nn.functional.sigmoid(logits)
    labels = torch.nn.functional.one_hot(labels, num_classes)
    labels = torch.tensor(labels, dtype=torch.float32)
    poly1 = labels * p + (1 - labels) * (1 - p)
    poly1_focal_loss = focal_loss + torch.mean(epsilon * torch.pow(1 - poly1, 2 + 1), dim=-1)
    return poly1_focal_loss
