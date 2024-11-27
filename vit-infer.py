import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification, ViTConfig
from datetime import datetime
from tqdm import tqdm  # 导入 tqdm 用于进度条显示

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
# 获取可用的 GPU 数量
num_gpu = torch.cuda.device_count()
print(f"可用的 GPU 数量: {num_gpu}")

# 定义数据集类
class ImageClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split('\t')
                self.img_labels.append((image_name, int(label)))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# 获取类别数量
def get_num_classes(annotations_file):
    labels = []
    with open(annotations_file, 'r') as f:
        for line in f:
            _, label = line.strip().split('\t')
            labels.append(int(label))
    num_classes = max(labels) + 1
    return num_classes

num_classes = get_num_classes('./annotations/train.txt')
print(f"类别数量: {num_classes}")

# 定义图像转换，包括数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    # 数据增强可以根据需要启用
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomGrayscale(p=0.2),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 的均值和标准差
    #                      std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

# 创建数据集和数据加载器
train_dataset = ImageClassificationDataset(
    annotations_file='./annotations/train_aug.txt',
    img_dir='./trainset_aug',
    transform=train_transform
)

val_dataset = ImageClassificationDataset(
    annotations_file='./annotations/val.txt',
    img_dir='./valset',
    transform=val_transform
)

# 设置 DataLoader 的 num_workers 数量适当调整，过高可能导致内存问题
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

# 定义本地模型路径
local_model_path = './vit'  # 你的模型文件所在的目录

# 加载配置文件
config = ViTConfig.from_pretrained(local_model_path)
config.num_labels = num_classes  # 设置类别数量

# 加载预训练的 ViT 模型
model = ViTForImageClassification.from_pretrained(local_model_path, config=config)

# 使用 DataParallel 如果有多个 GPU
if num_gpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)

# 定义要加载的最佳模型路径
best_model_file = './vit-best-models/best_model_epoch1_39.5890_mae.pth'
if os.path.exists(best_model_file):
    print(f"加载最佳模型权重：{best_model_file}")
    state_dict = torch.load(best_model_file, map_location=device)
    if num_gpu > 1:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print("最佳模型权重加载成功！")
else:
    print(f"指定的最佳模型文件不存在: {best_model_file}")
    exit(1)  # 如果模型不存在，退出程序

# 确保所有参数都可训练（虽然在推理阶段这不是必须的，但保留以防未来修改）
for param in model.parameters():
    param.requires_grad = True

# 定义 MAE 损失函数
mae_criterion = nn.L1Loss()

# 设置模型为评估模式
model.eval()

# 初始化统计变量
correct = 0
total = 0
total_mae = 0.0

with torch.no_grad():
    val_bar = tqdm(val_loader, desc="进行推理并计算准确率和 MAE", leave=False)
    for images, labels, _ in val_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images).logits
        _, preds = torch.max(outputs, 1)

        # 计算准确率
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 计算 MAE
        mae = mae_criterion(preds.float(), labels.float())
        total_mae += mae.item() * labels.size(0)  # 累计 MAE

        # 更新进度条信息
        current_acc = 100 * correct / total
        current_mae = total_mae / total
        val_bar.set_postfix(accuracy=f"{current_acc:.2f}%", mae=f"{current_mae:.4f}")

# 计算最终的准确率和 MAE
accuracy = 100 * correct / total
avg_mae = total_mae / total

print(f"验证集准确率: {accuracy:.2f}%")
print(f"验证集平均绝对误差 (MAE): {avg_mae:.4f}")
