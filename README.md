<h1 style="text-align: center;">8×8 像素矩阵手写数字识别</h1>

## 概述
本项目实现了一个基于卷积神经网络（CNN）的分类模型。该模型使用了Keras框架，并且设计了4个卷积层、最大池化层、全连接层等结构。该模型可以应用于图像分类任务，尤其适用于较小尺寸的灰度图像（例如8x8的图像）。

### 模型结构
1. **输入层**
   - 输入形状：(8, 8, 1)
   - 该模型的输入是一个8x8的灰度图像，单通道（1表示灰度图）。

2. **卷积层和池化层**
   - **第一层卷积层**：
     - 卷积核数量：32
     - 卷积核大小：(3, 3)
     - 激活函数：ReLU
     - 填充方式：same（保持输入与输出的大小相同）
     - 池化层：MaxPooling2D，池化大小为 (2, 2)，步长为 (1, 1)。
   
   - **第二层卷积层**：
     - 卷积核数量：64
     - 卷积核大小：(3, 3)
     - 激活函数：ReLU
     - 填充方式：same
     - 池化层：MaxPooling2D，池化大小为 (2, 2)，步长为 (1, 1)。
   
   - **第三层卷积层**：
     - 卷积核数量：128
     - 卷积核大小：(3, 3)
     - 激活函数：ReLU
     - 填充方式：same
     - 池化层：MaxPooling2D，池化大小为 (2, 2)，步长为 (1, 1)。
   
   - **第四层卷积层**：
     - 卷积核数量：256
     - 卷积核大小：(3, 3)
     - 激活函数：ReLU
     - 填充方式：same
     - 池化层：MaxPooling2D，池化大小为 (2, 2)，步长为 (1, 1)。

3. **展平层**
   - 将二维的卷积输出展平成一维，以便输入到全连接层。

4. **全连接层**
   - 神经元数量：256
   - 激活函数：ReLU

5. **输出层**
   - 神经元数量：10（适用于10分类任务）
   - 激活函数：Softmax（用于多分类任务）

### 编译和优化
- 优化器：Adam
- 损失函数：categorical_crossentropy
- 评估指标：accuracy（准确率）
### 项目展示

#### 界面
![](https://github.com/yilingxinghan/HandwrittenDigitRecognition/blob/master/resource/demonstration.png)

#### 识别效果
![](https://github.com/yilingxinghan/HandwrittenDigitRecognition/blob/master/resource/interface.png)


## 总结
该CNN模型适用于图像分类任务，并且使用Keras进行实现。模型包含多个卷积层和池化层，用于从输入图像中提取特征，最后通过全连接层进行分类。该模型可以通过提供的接口进行训练、评估、预测和保存。

## 作者
- 一零星寒
