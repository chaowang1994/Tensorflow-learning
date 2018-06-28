#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import os
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	
#写TFrecord文件
def _convrt_to_tfrecord():
   
    ''' write into tfrecord file '''
    if os.path.exists('minist.tfrecords'):
        os.remove('minist.tfrecords')

    writer = tf.python_io.TFRecordWriter('minist.tfrecords') # 创建.tfrecord文件，准备写入
 
    walkTest_tree = os.walk("./mnist") # mnist文件夹里是手写字体的图片，label取自图片名字的第一个数字

    for dirName, subDir, image_path in walkTest_tree:
       print(dirName)
       #print(subDir)
       #print(image_path) # mnist图片相对路径 例如 0_0.bmp

    for i in range(10000):

        mg = Image.open("mnist/"+image_path[i])
        mg = mg.resize((28, 28))
        img_raw = mg.tobytes()  
        example = tf.train.Example(features=tf.train.Features(
                feature={   
                'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[ (int)(image_path[i][0]) ]))    
                }))
        writer.write(example.SerializeToString()) 
    writer.close()

#读TFrecord文件
def _read_tfrecord(batch_size):
    filename_name = "minist.tfrecords"  #TFrecord文件名
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename_name], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
									 "img_raw": tf.FixedLenFeature([], tf.string), 
									 "label": tf.FixedLenFeature([], tf.int64)})						 
    image = tf.decode_raw(features["img_raw"], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32)

    labels = tf.cast(features["label"], tf.int32)
    labels = tf.one_hot(labels, 10)
    
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    image, label = tf.train.shuffle_batch([image, labels],
                                                          batch_size=batch_size, 
                                                          num_threads=4, 
                                                          capacity=10000,
                                                          min_after_dequeue=5000)
    return image, label
 
#对训练图像进行卷积运算
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
 
def conv2d(x,W):  
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def network(images, classes):
 
    #第一层卷积层
    W_conv1=weight_variable([5,5,1,32])
    b_conv1=bias_variable([32])
    h_conv1=tf.nn.relu(conv2d(images,W_conv1)+b_conv1)
    h_pool1=max_pool_2x2(h_conv1)

    #第二层卷积层
    W_conv2=weight_variable([5,5,32,64])
    b_conv2=bias_variable([64])
    h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2=max_pool_2x2(h_conv2)

    #全连接层，输出为1024维的向量
    W_fc1=weight_variable([7*7*64,1024])
    b_fc1=bias_variable([1024])
    h_pool2_flat=tf.reshape(h_pool2 ,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
 
    #把1024维的向量转换成10维，对应为10类
    W_fc2=weight_variable([1024,10])
    b_fc2=bias_variable([10])
    fc3l=tf.matmul(h_fc1, W_fc2)+b_fc2

    return fc3l

if __name__=='__main__':
    logs_train_dir = 'log_train'
    _convrt_to_tfrecord()
    train_batch, train_label_batch = _read_tfrecord(32)

    X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    Y = tf.placeholder(dtype=tf.int64, shape=[None, 10])
    print(X)
    print(Y)
    logits = network(X, 10) 

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(Y,1),tf.argmax(logits,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
  
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in np.arange(10000):
           if coord.should_stop():
               break
           train_images, train_labels = sess.run([train_batch, train_label_batch])
        
           _ , train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={X: train_images, Y : train_labels})
         
           if step % 50 == 0:
              print('Step %d, loss %f, acc %.2f%%' % (step, train_loss ,train_acc * 100.0))
              summary_str = sess.run(summary_op)
              train_writer.add_summary(summary_str, step)
             
        checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        
        coord.request_stop()
        coord.join(threads)
