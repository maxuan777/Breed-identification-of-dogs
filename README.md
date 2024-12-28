# Breed identification of dogs
 使用 EfficientNet 模型进行狗的品种分类
 
本项目通过使用 EfficientNet 模型进行狗的品种分类。我们使用了一个标准的数据集，该数据集包含了多种狗的品种及其相应的标签。我们将数据集进行预处理，训练模型，并使用该模型来预测图像中狗的品种。最后，结合 OpenCV 和 PIL 库，我们实现了图像识别结果的可视化。
在开始之前，首先需要设置必要的开发环境。确保安装以下Python库：
PyTorch：深度学习框架
torchvision：提供常用的深度学习模型和数据处理工具
PIL (Pillow)：图像处理库
opencv：用于图像显示和处理
