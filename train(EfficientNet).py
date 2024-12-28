
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet  # 导入EfficientNet
from sklearn.preprocessing import LabelEncoder
import copy

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
num_epochs = 50
batch_size = 32
learning_rate = 0.001

# 文件路径
train_dir = 'train'
labels_file = "labels.csv"

# 读取标签文件
labels_df = pd.read_csv(labels_file)
label_encoder = LabelEncoder()
labels_df['breed'] = label_encoder.fit_transform(labels_df['breed'])

# 自定义数据集类
class DogDataset(Dataset):
    def __init__(self, img_dir, labels_df=None, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform
        self.img_names = os.listdir(img_dir)

        if self.labels_df is not None:
            self.labels_df = self.labels_df.set_index('id')
            self.img_names = [img for img in self.img_names if 
                              img.split('.')[0] in self.labels_df.index]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels_df is not None:
            label = self.labels_df.loc[img_name.split('.')[0], 'breed']
            label = torch.tensor(label, dtype=torch.long)  # 确保标签为 LongTensor 类型
            return image, label
        else:
            return image, img_name

# 数据增强和预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用原始数据集
train_df = labels_df
val_df = labels_df

# 创建数据集和数据加载器
image_datasets = {
    'train': DogDataset(train_dir, train_df, data_transforms['train']),
    'val': DogDataset(train_dir, val_df, data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, 
                        shuffle=True, num_workers=0),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                      shuffle=False, num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = label_encoder.classes_

# 模型定义
model = EfficientNet.from_pretrained('efficientnet-b0')  # 使用EfficientNet-B0
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'第 {epoch+1} 轮/共 {num_epochs} 轮')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'最佳验证准确率: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    
    # 保存训练好的模型
    torch.save(model.state_dict(), 'trained_model-EfficientNet.pth')
    print("训练好的模型已保存")

    return model

# 训练模型
model = train_model(model, dataloaders, criterion, optimizer, num_epochs)
