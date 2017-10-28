'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, D, G):

        self.D = D   # discriminator
        self.G = G   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        self.D.add(Dense(256, input_shape = (784, ), activation = 'relu'))
        self.D.add(Dropout(0.4))
        self.D.add(Dense(1, activation = 'sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        self.G.add(Dense(512, input_dim=100, activation = 'relu'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Dropout(0.4))
        self.G.add(Dense(784, activation = 'sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self, exe=False):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        #optimizer = Adam(lr=0.0002, decay=6e-8)
        self.DM = Sequential()

        if exe==False:
            self.DM.add(self.discriminator())
        else:
            self.DM.add(self.D)

        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self, exe=False):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        #optimizer = Adam(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        if exe==False:
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
        else:
            self.AM.add(self.G)
            self.AM.add(self.D)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self, D=None, G=None):

        self.x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images
        self.DCGAN = DCGAN(D, G)

        if D==None:
            self.discriminator =  self.DCGAN.discriminator_model()
        else:
            self.discriminator =  self.DCGAN.discriminator_model(exe=True)

        if G==None:
            self.generator = self.DCGAN.generator()
            self.adversarial = self.DCGAN.adversarial_model()
        else:
            self.generator = G
            self.adversarial = self.DCGAN.adversarial_model(exe=True)



    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            self.discriminator.trainable = True
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            self.discriminator.trainable = False
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = './gan_tutor_output/mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "./gan_tutor_output/mnist_%d.png" % step
            images = self.generator.predict(noise).reshape(-1, 28, 28, 1)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :].reshape(-1, 28, 28, 1)

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [28, 28])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

def demo():
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=100)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)

def exe(dis, gen, steps=20000, save=1000):
    mnist_dcgan = MNIST_DCGAN(D=dis, G=gen)
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=steps, batch_size=256, save_interval=save)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)


if __name__ == '__main__':
    demo()
