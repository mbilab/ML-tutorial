#!/usr/bin/env python3

# see https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py

# for `_tkinter.TclError: no display name and no $DISPLAY environment variable`
#import matplotlib
#matplotlib.use('Agg')

from datetime import datetime
import h5py
from keras.layers import Activation, Dense, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Dropout, LeakyReLU
from keras.models import load_model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

class GAN:

    def __init__(self, image_width, image_height, image_channels, discriminator=None, generator=None, noise_size=100):
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.noise_size = noise_size
        self.D = discriminator if discriminator else self.default_discriminator()
        self.G = generator if generator else self.default_generator()
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

    # (W-F+2P)/S+1 #! what's this?
    def default_discriminator(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(self.image_width * self.image_height, )))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def default_generator(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.noise_size))
        model.add(Dense(self.image_width * self.image_height, activation='sigmoid')) #! self.channels?
        return model

    def discriminator_model(self):
        #optimizer = Adam(lr=0.0002, decay=6e-8)
        optimizer = RMSprop(lr=0.0002, decay=6e-8)

        model = Sequential()
        model.add(self.D) #! model = self.D?
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        return model

    def fit(self, x_train, batch_size=10000, callback=None, epochs=1):
        self.fit_start = datetime.now()
        self.training_step = 0
        for self.i_epoch in range(epochs):
            shuffled_x = shuffle(x_train)
            for i_batch in range(0, len(x_train), batch_size):
                self.training_step += 1

                batch = shuffled_x[i_batch:i_batch+batch_size]
                generated_batch = self.G.predict(self.noise(batch_size))
                x = np.concatenate((batch, generated_batch))
                y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
                self.DM.trainable = True
                self.d_loss = self.DM.train_on_batch(x, y)

                x = self.noise(batch_size)
                y = np.ones([batch_size, 1])
                self.DM.trainable = False
                self.a_loss = self.AM.train_on_batch(x, y)
                #x = self.noise(batch_size)
                #a_loss = self.AM.train_on_batch(x, y)

                if callback: callback()

    def fit_status(self):
        duration = datetime.now() - self.fit_start
        return "(%s) #%d: d_loss: %.3f, d_acc: %.3f, a_loss: %.3f, a_acc: %.3f" % \
                (duration, self.training_step, self.d_loss[0], self.d_loss[1], self.a_loss[0], self.a_loss[1])

    def noise(self, batch_size):
        return np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_size])

    def plot_images(self, cols=10, figsize=(15, 1.5), images=None, noise=10, save=None, title=None):
        if images is None:
            if isinstance(noise, int):
                noise = self.noise(noise)
            images = self.G.predict(noise).reshape(-1, self.image_width, self.image_height)

        plt.figure(figsize=figsize)
        rows = np.ceil(len(images) / cols)
        for i, v in enumerate(images):
            plt.subplot(rows, cols, i+1)
            plt.axis('off')
            plt.imshow(v.reshape(v.shape[0], v.shape[1]), cmap='gray')
        if title: plt.suptitle(title)
        if save:
            plt.savefig(save)
        else:
            plt.show()

class MNIST_DCGAN(GAN):

    def __init__(self, **kwargs):
        super(MNIST_DCGAN, self).__init__(28, 28, 1, **kwargs)

    def default_discriminator(self): # {{{
        model = Sequential()
        depth = 64
        dropout = 0.4

        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth = 64
        input_shape = (self.image_width, self.image_height, self.image_channels)
        model.add(Conv2D(depth*2, 5, strides=2, input_shape=input_shape, padding='same'))
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
        return model
    # }}}

    def default_generator(self): # {{{
        model = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7

        # in: 100
        # out: dim x dim x depth
        model.add(Dense(dim*dim*depth, input_dim=100))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((dim, dim, depth)))
        model.add(Dropout(dropout))

        # in: dim x dim x depth
        # out: 2*dim x 2*dim x depth/2
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        # out: 28 x 28 x 1 grayscale image, i.e. [0.0,1.0] per pixel
        model.add(Conv2DTranspose(1, 5, padding='same'))
        model.add(Activation('sigmoid'))
        return model
    # }}}

    def fit(self, *args, **kwargs):
        super(MNIST_DCGAN, self).fit(args[0].reshape(-1, 28, 28, 1), **kwargs)

def demo_dcgan():
    return load_model('./.prepared/mnist_dcgan.G.h5')

def demo_gan():
    return load_model('./.prepared/mnist_gan.G.h5')

def mnist_data():
    # download from aws
    path = get_file(
            'mnist.npz',
            'https://s3.amazonaws.com/img-datasets/mnist.npz',
            cache_dir='.',
            cache_subdir='.',
            file_hash='8a61469f7ea1b51cbae51d4f78837e45'
            )

    # load and normalize
    f = np.load(path)
    x_train, y_train = f['x_train'].reshape(-1, 784) / 255, to_categorical(f['y_train'], 10)
    x_test, y_test = f['x_test'].reshape(-1, 784) / 255, to_categorical(f['y_test'], 10)
    f.close()

    return x_train, y_train, x_test, y_test

def mnist_dcgan(): # {{{
    batch_size = 500
    epochs = 10

    def check():
        if gan.training_step % gan.steps_per_output: return
        gan.i_image += 1
        print(gan.fit_status())
        gan.plot_images(noise=check_noise, save='./mnist_dcgan/%02d.png' % (gan.i_image), title=gan.fit_status())

    gan = MNIST_DCGAN()
    check_noise = gan.noise(10)
    gan.i_image = 0
    steps = 60000 * epochs / batch_size # sample size = 60000
    gan.steps_per_output = steps / 10 # output 10 images
    gan.fit(x_train, batch_size=batch_size, callback=check, epochs=epochs)
    gan.D.save('./.prepared/mnist_dcgan.D.h5')
    gan.G.save('./.prepared/mnist_dcgan.G.h5')
    # }}}

def mnist_gan(): # {{{
    batch_size = 100
    epochs = 40

    def check():
        if gan.training_step % gan.steps_per_output: return
        gan.i_image += 1
        print(gan.fit_status())
        gan.plot_images(noise=check_noise, save='./mnist_gan/%02d.png' % (gan.i_image), title=gan.fit_status())

    gan = GAN(28, 28, 1)
    check_noise = gan.noise(10)
    gan.i_image = 0
    steps = 60000 * epochs / batch_size # sample size = 60000
    gan.steps_per_output = steps / 10 # output 10 images
    gan.fit(x_train, batch_size=batch_size, callback=check, epochs=epochs)
    gan.D.save('./.prepared/mnist_gan.D.h5')
    gan.G.save('./.prepared/mnist_gan.G.h5')
    # }}}

def test():
    batch_size = 6000
    epochs = 1

    def check():
        if gan.training_step % gan.steps_per_output: return
        gan.i_image += 1
        print(gan.fit_status())
        gan.plot_images(noise=check_noise, save='./mnist_gan/%02d.png' % (gan.i_image), title=gan.fit_status())

    gan = GAN(28, 28, 1)
    check_noise = gan.noise(10)
    gan.i_image = 0
    steps = 60000 * epochs / batch_size # sample size = 60000
    gan.steps_per_output = steps / 10 # output 10 images
    gan.fit(x_train, batch_size=batch_size, callback=check, epochs=epochs)

x_train, y_train, x_test, y_test = mnist_data()

if __name__ == '__main__':
    #mnist_gan()
    #mnist_dcgan()
    test()
