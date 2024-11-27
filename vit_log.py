import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification, ViTConfig
from torch.nn import HuberLoss
from datetime import datetime
from tqdm import tqdm  # 导入 tqdm 用于进度条显示
import logging  # 导入 logging 模块
import sys

# 配置 logging
log_filename = f"./vit-log/training_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)  # 同时输出到控制台
    ]
)

# 重定义 print 函数为 logging.info
def print(*args, **kwargs):
    logging.info(' '.join(map(str, args)))

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
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True)

# 定义本地模型路径
local_model_path = './vit'  # 你的模型文件所在的目录

# 加载配置文件
config = ViTConfig.from_pretrained(local_model_path)
config.num_labels = num_classes  # 设置类别数量

# 加载预训练的 ViT 模型
model = ViTForImageClassification.from_pretrained(local_model_path, config=config)

model = nn.DataParallel(model)

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

mae_criterion = nn.L1Loss()

huber_criterion = nn.SmoothL1Loss()
def combined_loss(outputs, labels, alpha=1.0):
    ce_loss = criterion(outputs, labels)  # 交叉熵损失
    preds = torch.argmax(outputs, dim=1).float()
    huber_loss = huber_criterion(preds, labels.float())  # Huber 损失
    return ce_loss + alpha * huber_loss  # alpha 是权重系数，可根据需要调整


# 使用 AdamW 优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4 * num_gpu, weight_decay=1e-2)

# 使用学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义要加载的最佳模型路径
continue_model_file = './vit-best-models/2024-11-27-epoch100-bt512/best_model_epoch80_2.80%.pth'
if os.path.exists(continue_model_file):
    print(f"加载最佳模型权重：{continue_model_file}")
    state_dict = torch.load(continue_model_file)
    if num_gpu > 1:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print("最佳模型权重加载成功！")
else:
    print(f"指定的最佳模型文件不存在: {continue_model_file}")
    # 根据需要，您可以选择退出程序或继续训练
    # exit(1)

# 确保所有参数都可训练
for param in model.parameters():
    param.requires_grad = True

# 定义解冻层的函数
def unfreeze_layers(model, layer_names):
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                if not param.requires_grad:
                    param.requires_grad = True
                    print(f"解冻层：{name}")

# 训练循环
num_epochs = 100
freeze_epochs = 5  # 每隔多少个 epoch 解冻一层
total_layers = config.num_hidden_layers  # ViT 模型的总层数
best_accuracy = 0.0  # 记录最佳验证准确率
best_mae = float('inf')  # 记录最佳验证 MAE，初始化为无穷大

best_model_dir = "./vit-best-models"
os.makedirs(best_model_dir, exist_ok=True)
best_model_file = os.path.join(best_model_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth")

for epoch in range(num_epochs):
    # 计算当前需要解冻的层数
    layers_to_unfreeze = min(total_layers, max(0, (epoch // freeze_epochs)))
    if layers_to_unfreeze > 0:
        layer_names = [f'encoder.layer.{total_layers - i - 1}' for i in range(layers_to_unfreeze)]
        print(f"第 {epoch+1} 个 epoch，解冻 {layers_to_unfreeze} 个层")
        unfreeze_layers(model, layer_names)
        # 更新优化器，只包含需要训练的参数
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5 * num_gpu, weight_decay=1e-2)
        # 重新定义学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练阶段
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False, file=sys.stdout)
    for images, labels, _ in train_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images).logits  # 获取模型输出
        # loss = criterion(outputs, labels)
        loss = combined_loss(outputs, labels)  # 使用联合损失

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=loss.item(), accuracy=f"{100 * correct / total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    total_mae = 0.0  # 初始化 MAE 累计
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False, file=sys.stdout)
    with torch.no_grad():
        for images, labels, _ in val_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 计算 MAE
            preds = torch.argmax(outputs, dim=1)
            mae = mae_criterion(preds.float(), labels.float())
            total_mae += mae.item() * labels.size(0)  # 累计 MAE

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_acc = 100 * correct / total
            val_bar.set_postfix(loss=loss.item(), accuracy=f"{current_acc:.2f}%")
    accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = total_mae / total  # 计算平均 MAE
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation MAE: {avg_val_mae:.4f}')

    # 更新学习率
    scheduler.step()
    # 检查是否为最佳 MAE，并保存模型
    if avg_val_mae < best_mae:
        best_mae = avg_val_mae
        best_model_file = os.path.join(best_model_dir, f"best_model_epoch{epoch+1}_{best_mae:.4f}_mae.pth")
        torch.save(model.module.state_dict() if num_gpu > 1 else model.state_dict(), best_model_file)
        print(f"最佳模型已保存，MAE: {best_mae:.4f}, 文件: {best_model_file}")
    # 检查是否为最佳准确率，并保存模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_file = os.path.join(best_model_dir, f"best_model_epoch{epoch+1}_{best_accuracy:.2f}%.pth")
        torch.save(model.module.state_dict(), best_model_file)
        print(f"最佳模型已保存，准确率: {best_accuracy:.2f}%")

print(f"训练完成。最佳验证准确率: {best_accuracy:.2f}%")

# 使用最佳模型进行预测并保存结果
# 加载最佳模型
best_model = ViTForImageClassification.from_pretrained(local_model_path, config=config)
# best_model = nn.DataParallel(best_model)
best_model = best_model.to(device)
# best_model.load_state_dict(torch.load('best_vit_model.pth'))
best_model.load_state_dict(torch.load(best_model_file))
best_model.eval()

all_predictions = []
all_image_names = []
with torch.no_grad():
    val_bar = tqdm(val_loader, desc="使用最佳模型保存预测结果", leave=False, file=sys.stdout)
    for images, labels, img_names in val_bar:
        images = images.to(device, non_blocking=True)
        outputs = best_model(images).logits
        _, preds = torch.max(outputs, 1)

        all_predictions.extend(preds.cpu().numpy())
        all_image_names.extend(img_names)

# 保存预测结果
pred_result_file = './pred_result.txt'
with open(pred_result_file, 'w') as f:
    for img_name, pred_label in zip(all_image_names, all_predictions):
        f.write(f"{img_name}\t{pred_label}\n")
print(f"预测结果已保存到 '{pred_result_file}'")
