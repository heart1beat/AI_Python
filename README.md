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
