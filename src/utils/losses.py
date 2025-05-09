import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BCEDiceLoss']


# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, input, target):
#         bce = F.binary_cross_entropy_with_logits(input, target)
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         target = target.view(num, -1)
#         intersection = (input * target)
#         dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
#         dice = 1 - dice.sum() / num
#         return 0.5 * bce + dice
    
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Binary Cross-Entropy Loss
        bce = F.binary_cross_entropy_with_logits(input, target)
        
        # Dice Loss
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = input.size(0)  # Batch size
        channels = input.size(1)  # Number of channels
        
        input = input.view(num, channels, -1)  # Reshape to (batch, channels, flattened spatial)
        target = target.view(num, channels, -1)
        
        intersection = (input * target).sum(-1)  # Sum over spatial dimensions
        dice = (2. * intersection + smooth) / (input.sum(-1) + target.sum(-1) + smooth)  # Per channel
        
        dice = 1 - dice.mean()  # Average Dice loss over channels
        
        # Combined loss
        return 0.5 * bce + dice

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
