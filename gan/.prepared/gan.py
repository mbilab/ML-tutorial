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
    model = load_model('./model.h5')
    dcgan = DCGAN(28, 28, 1, generator=model)
    pred = dcgan.G.predict(dcgan.noise(1, noise_size))
    print(pred.shape)
"""
    for i in range(10):
        file_name = folder + '/minist_%d.png' % i
        noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
        images = model.predict(noise).reshape(-1, 28, 28, 1)
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close('all')

    print('Output Image Saved!')
"""
class DCGAN(object):

    def __init__(self, image_width, image_height, image_channels, generator=None, discriminator=None, noise_size=100):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.D = self.default_discriminator() if discriminator == None else discriminator
        self.G = self.default_generator() if generator == None else generator
        self.DM = self.discriminator_model() # (D)
        self.AM = self.adversarial_model() # (G+D)
        self.noise_size = noise_size

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
        model.add(Dense(256, input_shape=(784, ), activation='relu')) #! self.image_width...
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def default_generator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.noise_size, activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Dropout(0.4))
        model.add(Dense(784, activation='sigmoid')) #! self.image_width...
        return model

    def discriminator_model(self):
        #optimizer = Adam(lr=0.0002, decay=6e-8)
        optimizer = RMSprop(lr=0.0002, decay=6e-8)

        model = Sequential()
        model.add(self.D)
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
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

    def noise(batch_size):
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
            plt.imshow(image, cmap='gray')
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
