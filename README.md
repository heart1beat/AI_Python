# 用TensorFlow建構的神經網路：MNIST數字辨識模型
南華大學跨領域學程-人工智慧期中報告
11124147 陳秉綜

##  將 TensorFlow 導入到程式碼

![image](https://github.com/heart1beat/AI_Python/blob/main/import_tensorflow.jpg)

## 加載Minist數據集

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
     
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step

## 建構機器學習模型

透過堆疊層來建構 tf.keras.Sequential 模型。

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
