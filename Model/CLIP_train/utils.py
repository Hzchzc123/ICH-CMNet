import os
import torch
from Model.CLIP_train.clip import CLIP

def clip_fusion(input):
    # 进行相似度的匹配
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备
    model = CLIP().to(DEVICE)  # 模型

    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    model.load_state_dict(torch.load(model_path), strict=False)   
    model.eval()  # 预测模式

    # 获取特定维度的大小
    batch_size = input.size(0)
    channels = input.size(1)
    height = input.size(2)
    width = input.size(3)

    image_in = []
    image_out = []

    for i in range(batch_size):
        image_in.append(input[i, :, :, :])  # 形状为 (channel, 64, 64)，有batch_size个

    print(image_in)

    for channel_image in image_in:
        other_images = [img for img in image_in if not torch.equal(img, channel_image)]
        img_emb = model.img_enc(torch.stack([channel_image], dim=0).to(DEVICE))
        other_img_embs = model.img_enc(torch.stack(other_images, dim=0).to(DEVICE))

        # 计算当前图片和其他图片的相似度
        logits = img_emb @ other_img_embs.T
        value, indexs = logits[0].topk(1)
        enhanced_image = (channel_image.to(DEVICE))  #  + value*0.1 将标量 value 扩展为与 channel_image 相同的形状
        image_out.append(enhanced_image)
    image_out_tensor = torch.stack(image_out, dim=0)

    image_in.clear()
    image_out.clear()

    return image_out_tensor
if __name__ == '__main__':
    img_x = torch.randn(32, 1, 64, 64)
    print(clip_fusion(img_x).shape)
