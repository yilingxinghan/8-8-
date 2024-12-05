from keras import layers, models
import numpy as np

class CNNModel(object):
    def __init__(self, path=None):
        if path:
            # 加载模型
            self.model = models.load_model(path)
            return
        
        # 创建模型
        self.model = models.Sequential()

        # 第一层卷积层
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # 第二层卷积层
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # 第三层卷积层
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # 第四层卷积层
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

        # 展平层
        self.model.add(layers.Flatten())

        # 全连接层
        self.model.add(layers.Dense(256, activation='relu'))

        # 输出层
        self.model.add(layers.Dense(10, activation='softmax'))

        # 编译模型
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def summary(self):
        self.model.summary()
    
    def train(self, x_train, y_train, x_test, y_test):
        # 训练模型
        history = self.model.fit(
            x_train, 
            y_train,
            batch_size=64,
            epochs=100,
            validation_data=(x_test, y_test)
        )
        return history

    def evaluate(self, x_test, y_test):
        # 评估模型
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_acc}")

    def predict_image(self, filepath):
        # 读取图像并转换为NumPy数组
        image = Image.open(filepath).convert('L')  # 将图像转换为灰度图
        image = image.resize((8, 8))
        image_array = img_to_array(image) / 255.0  # 将PIL图像转换为NumPy数组
        image_array = np.expand_dims(image_array, axis=0)  # 添加批次维度
        predictions = self.model.predict(image_array)
        # 获取预测结果的类别索引
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        # 假设你有一个类别标签列表
        class_labels = [str(i) for i in range(10)]
        # 获取预测结果的类别标签
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label

    def predict(self, x_test):
        # 使用模型进行预测
        predictions = self.model.predict(x_test, verbose=0)
        # 获取预测结果的类别索引
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        # 假设你有一个类别标签列表
        class_labels = [str(i) for i in range(10)]
        # 获取预测结果的类别标签
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label

    def save_model(self, filepath):
        # 保存模型
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        # 加载模型
        self.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")

if __name__ == '__main__':
    from keras.preprocessing.image import img_to_array # type: ignore
    from tensorflow.keras.utils import to_categorical # type: ignore
    from tensorflow.keras.datasets import mnist # type: ignore
    from PIL import Image
    import numpy as np

    param = 1 # 0为训练，1为推理

    if param == 0:

        def resize_image(images, target_size: tuple):
            # 创建一个新的数组用于存储缩小后的图片
            images_resized = np.zeros((images.shape[0], 8, 8), dtype=np.float32)

            # 遍历每张图片进行缩放
            for i in range(images.shape[0]):
                # 取出第i张28x28的图像
                img = Image.fromarray(images[i], mode='L')  # 'L'模式代表灰度图
                # 缩放到8x8
                img_resized = img.resize(target_size, Image.ANTIALIAS)  # 使用抗锯齿模式
                # 将缩放后的图片转换回numpy数组，并存储
                
                images_resized[i] = np.array(img_resized)
            
            return images_resized

        # 加载数据集（MNIST数据集会自动下载并存储在指定路径）
        (x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')

        # 预处理数据：将输入数据形状调整为 (样本数, 8, 8, 1)，并进行归一化
        x_train = resize_image(x_train, (8, 8)).astype('float32') / 255.0
        x_test = resize_image(x_test, (8, 8)).astype('float32') / 255.0

        # 将标签进行 one-hot 编码
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # 打印数据集的形状以确认
        print(f"x_train shape: {x_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        CNN = CNNModel()
        CNN.summary()
        CNN.train(x_train, y_train, x_test, y_test)
        CNN.evaluate(x_test, y_test)

        CNN.save_model('CNNModel.h5')
    
    elif param == 1:
        model = CNNModel('CNNModel.h5')
        model.summary()
        print(model.predict('image.jpg'))