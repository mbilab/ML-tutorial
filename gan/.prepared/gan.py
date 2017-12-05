'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #! necessary?

from tensorflow.examples.tutorials.mnist import input_data

from sklearn.utils import shuffle
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.models import load_model, Sequential
from keras.optimizers import Adam, RMSprop

def demo(noise_size):
    model = load_model('./.prepared/model.h5')
    dcgan = DCGAN(image_width=28, image_height=28, image_channels=1)
    images = model.predict(dcgan.noise(noise_size))

    plt.figure(figsize=(8,8))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    #plt.savefig(file_name)
    #plt.close('all')
    #print('Output Image Saved!')

def exe(images, G, D, batch_size, noise_size):
    dcgan = DCGAN(image_width=28, image_height=28, image_channels=1, generator=G, discriminator=D, noise_size=noise_size)
    dcgan.fit(images=images)


class DCGAN(object):

    def __init__(self, image_width, image_height, image_channels, generator=None, discriminator=None, noise_size=100):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.noise_size = noise_size
        self.D = self.default_discriminator() if discriminator == None else discriminator
        self.G = self.default_generator() if generator == None else generator
        self.DM = self.discriminator_model() # (D)
        self.AM = self.adversarial_model() # (G+D)

    def adversarial_model(self):
        #optimizer = Adam(lr=0.0001, decay=3e-8)
        optimizer = RMSprop(lr=0.0001, decay=3e-8)

        model = Sequential()
        model.add(self.G)
        model.add(self.D)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        return model

    # (W−F+2P)/S+1 #! what's this?
    def default_discriminator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.image_width*self.image_height, ), activation='relu')) #! self.image_width...
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def default_generator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.noise_size, activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Dropout(0.4))
        model.add(Dense(self.image_width*self.image_height, activation='sigmoid')) #! self.image_width...
        return model

    def discriminator_model(self):
        #optimizer = Adam(lr=0.0002, decay=6e-8)
        optimizer = RMSprop(lr=0.0002, decay=6e-8)

        model = Sequential()
        model.add(self.D)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        return model

    def nice_generator():
        model = sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # in: 100
        # out: dim x dim x depth
        model.add(Dense(dim*dim*depth, input_dim=100))
        model.add(Batchnormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((dim, dim, depth)))
        model.add(Dropout(dropout))

        # in: dim x dim x depth
        # out: 2*dim x 2*dim x depth/2
        model.add(Upsampling2d())
        model.add(Conv2dtranspose(int(depth/2), 5, padding='same'))
        model.add(Batchnormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Upsampling2d())
        model.add(Conv2dtranspose(int(depth/4), 5, padding='same'))
        model.add(Batchnormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2dtranspose(int(depth/8), 5, padding='same'))
        model.add(Batchnormalization(momentum=0.9))
        model.add(Activation('relu'))

        # out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        model.add(Conv2dtranspose(1, 5, padding='same'))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def nice_discriminator():
        model = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.image_width, self.image_height, self.image_channels)
        model.add(Conv2D(depth*2, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth*1, 5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(depth*1, 5, strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        # Out: 1-dim probability
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        return model

    def fit(self, images, batch_size=256, epochs=10, plot_interval=0):
        step = len(x_train)//batch_size
        for i in range(epochs * step):
            k = i % step

            if k == 0:
                self.x_train = shuffle(self.x_train)

            batch = self.x_train[0*k:batch_size*k]
            #batch = self.x_train[np.random.randint(0, self.images.shape[0], size=batch_size), :]
            noise = self.noise(batch_size)
            fake_batch = self.G.predict(noise)
            x = np.concatenate((batch, fake_batch))
            y = np.concatenate(np.ones([batch_size, 1], np.zeros([batch_size, 0])))
            self.DM.trainable = True
            d_loss = self.DM.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = self.noise(batch_size)
            self.DM.trainable = False
            a_loss = self.AM.train_on_batch(noise, y)
            print("%d: D: loss: %f, acc: %f, A: loss: %f, acc: %f" % (i,
                d_loss[0], d_loss[1],
                a_loss[0], a_loss[1]
                ))

            if 0 == plot_interval | (i + 1) % plot_interval:
                continue
            #self.plot_images(output, save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))

    def noise(self, batch_size):
        return np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_size])

    def plot_images(self, images=None, noise=None, figsize=(20, 2), cols=10):
        if images is None:
            if isinstance(noise, int):
                noise = self.noise(noise)
            images = self.generator.predict(noise).reshape(-1, self.image_width, self.image_height, self.channels)

        plt.figure(figsize=figsize)
        rows = np.ceil(len(images) / 4)
        for i in range(len(images)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
        #plt.savefig(filename)
        #plt.close('all')
        plt.show()

    def save_g_model(self):
        self.G.save('path')

def test():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)

    dcgan = DCGAN(28, 28, 1)
    check_noise = dcgan.noise(10)
    dcgan.fit(minst.train.images, callback=check_images)

    def check_images():
        dcgan.plot_image(check_noise)


if __name__ == '__main__':
    demo(100)
