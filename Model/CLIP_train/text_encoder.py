from torch import nn
import torch
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=16, embedding_dim=16)  # 16个词汇，每个词汇16维
        self.dense1 = nn.Linear(in_features=16, out_features=64)
        self.dense2 = nn.Linear(in_features=64, out_features=16)
        self.wt = nn.Linear(in_features=16, out_features=8)  # 输出8维
        self.ln = nn.LayerNorm(8)  # LayerNorm处理8维的向量

    def forward(self, x):
        x = self.emb(x)  # 转换成16维嵌入
        x = F.relu(self.dense1(x))  # 第一个全连接层
        x = F.relu(self.dense2(x))  # 第二个全连接层
        x = self.wt(x)  # 转换为8维
        x = self.ln(x)  # 层归一化

        # 这里去掉多余的维度 (如果你只关心样本的嵌入表示)
        x = x.squeeze(1)  # 去掉第二维（seq_length = 1）

        return x


if __name__ == '__main__':
    text_encoder = TextEncoder()

    # 输入形状为 (16, 1)，表示16个样本，每个样本有1个词汇（词汇的索引）
    x=torch.randint(0,16,(16,1))
    print(x)
    # 输出
    y = text_encoder(x)  # 输入到TextEncoder
    print(y)  # 打印输出形状，应该是 (16, 8)，表示16个样本，每个样本8维的表示
