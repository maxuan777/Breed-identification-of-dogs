import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载狗的品种名称
with open('class_names.txt', 'r', encoding='utf-8') as file:
    class_names = [line.strip() for line in file.readlines()]

# 加载预训练的模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval()

# 预测狗的品种
def predict_dog_breed(image_path, model, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_breed = class_names[predicted.item()]
    return predicted_breed

# 显示图片并在左上角显示狗的品种名称
def show_image_with_breed(image_path, breed):
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Error: File does not exist at {image_path}")
        return

    # 尝试读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # 将 OpenCV 图像转换为 PIL 图像
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 使用 PIL 绘制文本
    draw = ImageDraw.Draw(image_pil)
    # 字体文件路径
    font_path = 'simhei.ttf'  # 请确保该路径正确
    font = ImageFont.truetype(font_path, 30)
    
    # 绘制中文文本
    draw.text((10, 10), breed, font=font, fill=(255, 0, 0))

    # 将 PIL 图像转换回 OpenCV 图像
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # 显示图片
    cv2.namedWindow("Dog Breed Prediction",0);
    cv2.imshow('Dog Breed Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 预测示例图像
image_path = "test\\1b04182b865401f6f3e303c33d0f41a6.jpg" 
predicted_breed = predict_dog_breed(image_path, model, data_transforms, class_names)
print(f"预测的狗的品种是: {predicted_breed}")

# 显示图片并在左上角显示狗的品种名称
show_image_with_breed(image_path, predicted_breed)
