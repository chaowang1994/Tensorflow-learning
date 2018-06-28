tensorflow version : '1.8.0'

## 概述：
关于tensorflow读取数据，官网给出了三种方法：

1、供给数据：在tensorflow程序运行的每一步，让python代码来供给数据

2、从文件读取数据：建立输入管线从文件中读取数据

3、预加载数据：如果数据量不太大，可以在程序中定义常量或者变量来保存所有的数据。
  
具体实现步骤参考这些：
https://www.jianshu.com/p/78467f297ab5
https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

说明：

1. mnist_show_digits.py  构建tfrecord格式的mnist，并显示出从TFrecord文件中读出来的图片
2. tfrecord_mnist_cnn.py 从mnist文件夹里读取图片构建tfrecord再用CNN进行训练
 