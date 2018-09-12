import os
import math
import json
import tensorflow as tf
from PIL import Image
from functools import partial
 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
def get_tf_dataset(dataset_text_file,batch_size=64, channels=3,crop_size=[28,28],shuffle_size=200,augmentation=False):
    def aug_1(image):
        image = tf.image.random_brightness(image, max_delta=2. / 255.)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        return image
 
    def aug_2(image):
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_brightness(image, max_delta=2. / 255.)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        return image
 
    def aug_3(image):
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_brightness(image, max_delta=2. / 255.)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        return image
 
    def aug_4(image):
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_saturation(image, lower=0.01, upper=0.05)
        image = tf.image.random_contrast(image, lower=0.01, upper=0.05)
        image = tf.image.random_brightness(image, max_delta=2. / 255.)
        return image
 
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
        image = tf.image.resize_images(image_decoded, crop_size)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        if augmentation:
            #tensorflow1.7支持tf.contrib.image.rotate
            angle = tf.reshape(tf.random_uniform([1], -math.pi/12, math.pi/12, tf.float32), [])
            image = tf.contrib.image.rotate(image, angle)
            #image = tf.image.random_flip_left_right(image)
 
            #image = tf.random_crop(image, [crop_size, crop_size, 3])
            
            p1 = partial(aug_1, image)
            p2 = partial(aug_2, image)
            p3 = partial(aug_3, image)
            p4 = partial(aug_4, image)
 
            k = tf.reshape(tf.random_uniform([1], 0, 4, tf.int32), [])
            image = tf.case([(tf.equal(k, 0), p1),
                             (tf.equal(k, 1), p2),
                             (tf.equal(k, 2), p3),
                             (tf.equal(k, 3), p4)],
                            default=p1,
                            exclusive=True)
        
        
        return image, label
 
    def read_labeled_image_list(dataset_text_file):
        filenames=[]
        labels=[]
        with open(dataset_text_file,"r",encoding="utf-8") as f_l:
            filenames_lables=f_l.readlines()
        for filename_lable in filenames_lables:
            filenames.append(filename_lable.split(" ")[0])
            labels.append(int(filename_lable.split(" ")[1].strip("\n")))
        return filenames,labels
 
    filenames, labels = read_labeled_image_list(dataset_text_file)
 
    filenames = tf.constant(filenames, name='filename_list')
    labels = tf.constant(labels, name='label_list')
 
    #tensorflow1.3:tf.contrib.data.Dataset.from_tensor_slices
    #tensorflow1.4+:tf.data.Dataset.from_tensor_slices
    dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat()
 
    return dataset
 
 
def train():
    #network
    batch_size = 64
    inputs = tf.placeholder(tf.float32, [None, 28, 28, 3], name='input')
    conv1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=(3, 3), padding="same", activation=None)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=None)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
 
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
    fc1 = tf.layers.dense(pool2_flat, 500, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 10, activation=tf.nn.relu)
    y_out = tf.nn.softmax(fc2,name='output')
    result = tf.argmax(y_out, 1,name='prediction')
 
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_mean(y_ * tf.log(y_out))  # 计算交叉熵
 
    learning_rate=1e-3
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))  # 判断预测标签和实际标签是否匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
    dataset = get_tf_dataset(dataset_text_file="./train.txt",batch_size=64)
    iterator = dataset.make_one_shot_iterator()
    img_batch, label_batch = iterator.get_next()
 
    init = tf.global_variables_initializer()
    
    keys_placeholder = tf.placeholder("float")
    keys = tf.identity(keys_placeholder)
    tf.add_to_collection("input", json.dumps({'key': keys_placeholder.name, 'features': inputs.name}))
    tf.add_to_collection('output', json.dumps({'key': keys.name, 'softmax': y_out.name, 'prediction': result.name}))
    with tf.Session() as session:
        session.run(init)
        threads = tf.train.start_queue_runners()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        for i in range(400):
            img_batch_i, label_batch_i = session.run([img_batch, tf.one_hot(label_batch, depth=10)])
 
            feed = {inputs: img_batch_i, y_: label_batch_i}
            loss,_,acc=session.run([cross_entropy,train_step,accuracy], feed_dict=feed)
 
            print("step%d loss:%f accuracy:%F"%(i,loss,acc))
            if i>100:
                learning_rate=learning_rate*0.1
        saver.save(session, "./save/mnist.ckpt")
 
 
if __name__=="__main__":
    train()
