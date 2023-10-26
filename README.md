# 用TensorFlow建構的神經網路：MNIST數字辨識模型
南華大學跨領域學程-人工智慧期中報告
11124147 陳秉綜

##  將 TensorFlow 導入到程式碼

![image](https://github.com/heart1beat/AI_Python/blob/main/import_tensorflow.jpg)

## 加載Minist數據集

![image](https://github.com/heart1beat/AI_Python/blob/main/mnist.jpg)

## 建構機器學習模型

透過堆疊層來建構 tf.keras.Sequential 模型。

![image](https://github.com/heart1beat/AI_Python/blob/main/model.jpg)

對於每個樣本，模型都會傳回一個包含 logits 或 log-odds 分數的向量，每個類別一個。

![image](https://github.com/heart1beat/AI_Python/blob/main/predictions.jpg)

tf.nn.softmax 函數將這些 logits 轉換為每個類別的機率：

![image](https://github.com/heart1beat/AI_Python/blob/main/softmax.jpg)

使用 losses.SparseCategoricalCrossentropy 為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每個樣本傳回一個標量損失。

![image](https://github.com/heart1beat/AI_Python/blob/main/define_loss_fn.jpg)

此損失等於 true 類別的負對數機率：如果模型確定類別正確，則損失為零。

這個未經訓練的模型給出的機率接近隨機（每個類別為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3。

![image](https://github.com/heart1beat/AI_Python/blob/main/calculate_loss_fn.jpg)

在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 
將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。

![image](https://github.com/heart1beat/AI_Python/blob/main/model_compile.jpg)

## 訓練並評估模型

使用 Model.fit 方法調整您的模型參數並最小化損失：

![image](https://github.com/heart1beat/AI_Python/blob/main/model_fit.jpg)

Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。

![image](https://github.com/heart1beat/AI_Python/blob/main/model_evaluate.jpg)

現在，這個照片分類器的準確度已經達到 98%。

如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：

![image](https://github.com/heart1beat/AI_Python/blob/main/create_probability_model.jpg)

![image](https://github.com/heart1beat/AI_Python/blob/main/use_probability_model.jpg)

## 參考文章:
https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
     
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
