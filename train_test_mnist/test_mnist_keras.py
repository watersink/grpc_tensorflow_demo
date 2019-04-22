
import cv2
import os

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




class Mnist_test(object):
    def __init__(self):
        self.network = Network()
        self.width = self.network.input_shape[0]
        self.height = self.network.input_shape[1]

        self.model_path=('./save/mnist_epoch99_valacc0.98_valloss0.05.hdf5')
        self.model=self.load_model()

    def load_model(self):
        model = self.network.character_network()
        model.load_weights(self.model_path)
        return model

    def test_mnist_images(self, gendir):
        rightnum = 0
        allnum = 0

        dirs = os.listdir(gendir)
        for i in range(0, len(dirs)):
            imgdir = gendir + dirs[i] + '/'
            picnames = os.listdir(imgdir)
            for j in range(0, len(picnames)):
                img = cv2.imread(imgdir + picnames[j])
                resized_img = cv2.resize(img, (self.width, self.width), interpolation=cv2.INTER_AREA)
                image = resized_img.reshape(1, self.width, self.width, 3)
                image = image.astype('float32') / 255.0
                list_of_list = self.model.predict(image, batch_size=1, verbose=1)
                classnum = list_of_list[0].tolist().index(max(list_of_list[0]))
                if classnum == i:
                    rightnum = rightnum + 1
                allnum = allnum + 1
        precision = rightnum / allnum
        print('TEST ACC：%f' % (precision))

if __name__=='__main__':
        m_test=Mnist_test()
        gendir = './MNIST/testimage/'
        m_test.test_mnist_images(gendir)