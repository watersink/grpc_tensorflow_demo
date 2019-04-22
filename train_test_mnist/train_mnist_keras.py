import cv2
import numpy as np
import h5py
import os, sys
import keras
import random
import time

from sklearn.model_selection import train_test_split
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise, GaussianDropout, BatchNormalization
from keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class Network(object):
    def __init__(self):
        self.input_shape = (28, 28, 3)
        self.num_classes = 10

    def character_network(self):
        # lenet-5   28*28
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.35))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        return model




class Model_train(object):
    def __init__(self):
        self.network = Network()
        self.model = self.network.character_network()
        self.img_size = (self.network.input_shape[:2])
        self.num_classes = self.network.num_classes
        self.test_size = 0.15
        self.imagepath = "./MNIST/trainimage/"
        self.train_batch_size = 128
        self.val_batch_size = int(self.train_batch_size * self.test_size)
        self.learningrate = 0.001
        self.epoch = 201

    def train_network(self):
        images_label_list_train, images_label_list_val = self.images_shuffle()
        # rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-8, decay = 0.0)
        sgd = SGD(lr=self.learningrate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        #X_train, X_test, y_train, y_test = self.get_dataset()
        # self.model.fit(X_train, y_train, epochs=20, batch_size=100)
        print("begin")
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath='mnist_epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5', monitor='val_acc',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=50)
        reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.model.fit_generator(self.generate_train_data(images_label_list_train), \
                                 samples_per_epoch=self.train_batch_size, \
                                 nb_epoch=self.epoch, \
                                 validation_data=self.generate_val_data(images_label_list_val), \
                                 nb_val_samples=self.val_batch_size, \
                                 verbose=1, nb_worker=1, \
                                 callbacks=[checkpointer,reducelr])

        # loss, accuracy = self.model.evaluate(X_test, y_test)
        # print('loss:%f accuracy:%f' % (loss, accuracy))
        # self.model.save('cr_deepocr.h5', overwrite=True)

    def images_shuffle(self):
        images_label_list_train = []
        images_label_list_val = []
        dirs = os.listdir(self.imagepath)
        for i in range(0, len(dirs)):
            imgdir = self.imagepath + dirs[i] + '/'
            picnames = os.listdir(imgdir)
            for j in range(0, len(picnames)):
                if j % int(len(picnames) / (len(picnames) * self.test_size)) != 0:
                    images_label_list_train.append([imgdir + picnames[j], int(dirs[i])])
                else:
                    images_label_list_val.append([imgdir + picnames[j], int(dirs[i])])
        # random.shuffle(images_label_list_train)
        # random.shuffle(images_label_list_val)
        # print(len(images_label_list_train))
        return (images_label_list_train, images_label_list_val)

    def generate_train_data(self, images_label_list_train):
        while True:
            random.shuffle(images_label_list_train)
            X_image = []
            Y_label = []
            count = 0
            for image in images_label_list_train:
                img = cv2.imread(image[0])
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                X_image.append(img)
                Y_label.append(image[1])
                count += 1
                if count == self.train_batch_size:
                    count = 0
                    label_train = keras.utils.to_categorical(np.asarray(Y_label), self.num_classes)
                    yield (np.asarray(X_image), label_train)
                    X_image = []
                    Y_label = []

    def generate_val_data(self, images_label_list_val):
        while True:
            random.shuffle(images_label_list_val)
            X_image = []
            Y_label = []
            count = 0
            for image in images_label_list_val:
                img = cv2.imread(image[0])
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                X_image.append(img)
                Y_label.append(image[1])
                count += 1
                if count == self.val_batch_size:
                    count = 0
                    label_val = keras.utils.to_categorical(np.asarray(Y_label), self.num_classes)
                    yield (np.array(X_image), label_val)
                    X_image = []
                    Y_label = []


if __name__ == '__main__':
    model = Model_train()
    model.train_network()
