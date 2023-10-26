# 用TensorFlow建構的神經網路：MNIST數字辨識模型
南華大學跨領域學程-人工智慧期中報告
11124147 陳秉綜

##  將 TensorFlow 導入到程式碼

```python
import tensorflow as tf
```

## 加載Minist數據集

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
     
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
```

## 建構機器學習模型

透過堆疊層來建構 tf.keras.Sequential 模型。

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

對於每個樣本，模型都會傳回一個包含 logits 或 log-odds 分數的向量，每個類別一個。

```python
predictions = model(x_train[:1]).numpy()
predictions
     
array([[-0.4545507 ,  0.98449874,  0.28289455,  0.50742406, -0.35388675,
        -0.6623234 , -0.6642075 ,  0.36214182,  0.2607504 , -0.10538845]],
      dtype=float32)
```

tf.nn.softmax 函數將這些 logits 轉換為每個類別的機率：

```python
tf.nn.softmax(predictions).numpy()
     
array([[0.05440999, 0.22942983, 0.11374886, 0.14238329, 0.06017228,
        0.04420222, 0.04411902, 0.12312996, 0.11125767, 0.0771468 ]],
      dtype=float32)
```

使用 losses.SparseCategoricalCrossentropy 為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每個樣本傳回一個標量損失。

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

此損失等於 true 類別的負對數機率：如果模型確定類別正確，則損失為零。

這個未經訓練的模型給出的機率接近隨機（每個類別為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3。

```python
loss_fn(y_train[:1], predictions).numpy()
     
3.1189804
```

在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 
將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
     
```

## 訓練並評估模型

使用 Model.fit 方法調整您的模型參數並最小化損失：

```python
model.fit(x_train, y_train, epochs=5)
     
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.3002 - accuracy: 0.9131
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1454 - accuracy: 0.9562
Epoch 3/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.1085 - accuracy: 0.9671
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0897 - accuracy: 0.9727
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0748 - accuracy: 0.9766
<keras.src.callbacks.History at 0x7a15a237be50>
```

Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。

```python
model.evaluate(x_test,  y_test, verbose=2)
     
313/313 - 1s - loss: 0.0730 - accuracy: 0.9768 - 1s/epoch - 4ms/step
[0.07296817004680634, 0.9768000245094299]
```

現在，這個照片分類器的準確度已經接近 98%。

如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

```python

probability_model(x_test[:5])
     
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[3.1357499e-07, 1.7557096e-09, 1.3425333e-06, 3.1300344e-05,
        3.1137037e-11, 1.4660566e-07, 4.0756773e-15, 9.9996185e-01,
        4.1396987e-07, 4.6358919e-06],
       [4.8699695e-07, 5.7306176e-04, 9.9924350e-01, 1.2119730e-04,
        4.4787817e-13, 1.9157314e-07, 2.3823028e-05, 6.7813488e-10,
        3.7788181e-05, 3.9807841e-13],
       [2.0344415e-07, 9.9618572e-01, 1.0719226e-04, 9.6817284e-06,
        1.4588626e-05, 1.2844469e-05, 1.1701437e-05, 2.8729190e-03,
        7.8392879e-04, 1.2016635e-06],
       [9.9992836e-01, 2.1502304e-10, 4.8260958e-05, 7.0681716e-07,
        5.0750573e-06, 2.8021424e-07, 5.6977478e-06, 7.3169040e-06,
        1.6803372e-07, 4.0014793e-06],
       [3.1008385e-06, 6.3397568e-12, 7.0007246e-07, 7.8687123e-10,
        9.9846661e-01, 1.2702397e-07, 3.1732424e-07, 7.3298710e-05,
        4.8488968e-08, 1.4557794e-03]], dtype=float32)>
```

## 參考文章:
https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn
x_train, x_test = x_train / 255.0, x_test / 255.0
     
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
