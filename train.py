import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from dataloader.dataloader import DataLoader, MultipleDatasets
from Model.net import Net
import csv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model.Loss_function import SegmentationLoss

# Dice损失
def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    running_dice = 0.0
    running_jaccard = 0.0
    running_hd95 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_count = 0

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清除之前的梯度

        # 前向传播
        outputs = model(images, labels)
        outputs = torch.sigmoid(outputs)  # 分割结果通常会通过sigmoid激活

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播

        # 梯度裁剪，如果需要的话
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()  # 更新参数

        # 计算评价指标
        dice = dice_coeff(outputs, labels)
        jaccard = jaccard_coeff(outputs, labels)
        hd95 = hausdorff_distance(outputs, labels)
        precision, recall = precision_recall(outputs, labels)

        # 更新统计量
        running_loss += loss.item()
        running_dice += dice
        running_jaccard += jaccard
        running_hd95 += hd95
        total_precision += precision
        total_recall += recall
        total_count += 1

    # 计算平均损失和指标
    avg_loss = running_loss / total_count
    avg_dice = running_dice / total_count
    avg_jaccard = running_jaccard / total_count
    avg_hd95 = running_hd95 / total_count
    avg_precision = total_precision / total_count
    avg_recall = total_recall / total_count

    print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Jaccard: {avg_jaccard:.4f}, HD95: {avg_hd95:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # 保存训练日志到文本文件
    with open("training_log.txt", "a") as log_file:
        log_file.write(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, Jaccard: {avg_jaccard:.4f}, HD95: {avg_hd95:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}\n")

    # 保存训练日志到CSV文件
    with open("training_log.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:  # 只有在文件为空时才写入表头
            writer.writerow(['epoch', 'loss', 'dice', 'jaccard', 'hd95', 'precision', 'recall'])
        writer.writerow([epoch, avg_loss, avg_dice, avg_jaccard, avg_hd95, avg_precision, avg_recall])

    # 保存模型权重
    # torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    return avg_loss, avg_dice, avg_jaccard, avg_hd95, avg_precision, avg_recall

def dice_coeff(pred, target):
    pred = pred > 0.5  # 二值化
    target = target > 0.5
    intersection = torch.sum(pred * target)
    return 2 * intersection / (torch.sum(pred) + torch.sum(target) + 1e-6)

def jaccard_coeff(pred, target):
    pred = pred > 0.5
    target = target > 0.5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return intersection / (union + 1e-6)

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

def precision_recall(pred, target):
    pred = pred.detach().cpu().numpy().squeeze() > 0.5
    target = target.cpu().numpy().squeeze() > 0.5

    precision, recall, _ = precision_recall_curve(target.flatten(), pred.flatten())
    avg_precision = precision.mean()
    avg_recall = recall.mean()

    return avg_precision, avg_recall

# 检查是否有多张 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

criterion = SegmentationLoss()

dataset = MultipleDatasets(root_dir=r"./dataloader/dataloaders_test")
dataloader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=4)

num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    train_loss, train_dice, train_jaccard, train_hd95, train_precision, train_recall = train(
        model, dataloader, criterion, optimizer, device, epoch
    )

# 保存最后一个epoch的数据
final_epoch = num_epochs
final_train_loss, final_train_dice, final_train_jaccard, final_train_hd95, final_train_precision, final_train_recall = train_loss, train_dice, train_jaccard, train_hd95, train_precision, train_recall

# 保存最后epoch的模型权重
torch.save(model.state_dict(), f"final_model_epoch{final_epoch}.pth")
