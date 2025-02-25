import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import os

from Model.CLIP_train.feature_clip import feature_clip
from dataloader import dataloader

# 使用 torch.device 来转换为设备对象
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

model = feature_clip(16).to(DEVICE)

dataset = dataloader.MultipleDatasets(
    root_dir=r"../../dataloader/dataloaders")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

'''
    训练模型
'''
ITER_BATCH_COUNT = 100  # 迭代次数
BATCH_SIZE = 16  # 从batch内选出10个不一样的数字
TARGET_COUNT = 15  # 共10种数字

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # 降低 workers 数量

for i in range(ITER_BATCH_COUNT):
    for imgs, _, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs, labels)  # DataParallel 会自动分配到多个 GPU 上
        targets = torch.arange(0, TARGET_COUNT + 1).to(DEVICE)
        logits = logits[:-9]  # 截取 logits 的最后 9 个样本
        targets = targets[:-9]  # 截取 targets 的最后 9 个样本
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iter:{},loss:{}'.format(i, loss))
        torch.save(model.state_dict(), '.model.pth')
        os.replace('.model.pth', '.model.pth')
