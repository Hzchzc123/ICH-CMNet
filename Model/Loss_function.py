aux | grrefrom torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# 定义Dice损失
def dice_loss(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# 定义二进制交叉熵损失
def bce_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return bce

def hausdorff_distance(pred, target):
    pred = pred.detach().cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    # 获取非零像素的坐标
    pred_coords = np.column_stack(np.where(pred > 0.5))
    target_coords = np.column_stack(np.where(target > 0.5))

    # 计算95%的Hausdorff距离
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return 0

    dist_pred_to_target = directed_hausdorff(pred_coords, target_coords)[0]
    dist_target_to_pred = directed_hausdorff(target_coords, pred_coords)[0]

    return max(dist_pred_to_target, dist_target_to_pred)

# 定义一个简化的Hausdorff距离损失
def hausdorff_loss(pred, target):
    # 注意：这里的实现是一个简化的版本，用于说明如何构造损失
    # 真实的Hausdorff距离是不可微分的，因此不能直接用作损失函数
    pred = torch.sigmoid(pred)
    hd95 = hausdorff_distance(pred, target)  # 假设这是计算Hausdorff距离的函数
    return hd95

class SegmentationLoss(nn.Module):
    def __init__(self,alpha=1,beta=1,gamma=1):
        super(SegmentationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        dice = dice_loss(pred, target)
        bce = bce_loss(pred, target)
        hausdorff = hausdorff_loss(pred, target)
        return self.alpha * dice + self.beta * bce + self.gamma * hausdorff

