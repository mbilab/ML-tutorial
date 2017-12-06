'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.models import load_model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def demo(sample_size):
    model = load_model('./.prepared/model.h5')
    dcgan = DCGAN(image_width=28, image_height=28, image_channels=1)
    images = model.predict(dcgan.noise(sample_size))

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

class GAN:

    def __init__(self, image_width, image_height, image_channels, generator=None, discriminator=None, noise_size=100):
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

    # (W−F+2P)/S+1 #! what's this?
    def default_discriminator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=(self.image_width * self.image_height, ), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def default_generator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.noise_size, activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Dropout(0.4))
        model.add(Dense(self.image_width * self.image_height, activation='sigmoid')) #! self.channels?
        return model

    def discriminator_model(self):
        #optimizer = Adam(lr=0.0002, decay=6e-8)
        optimizer = RMSprop(lr=0.0002, decay=6e-8)

        model = Sequential()
        model.add(self.D) #! model = self.D?
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        return model

    def fit(self, x_train, batch_size=10000, epochs=1):
        training_step = 0
        for i_epoch in range(epochs):
            shuffled_x = shuffle(x_train)
            for i_batch in range(0, len(x_train), batch_size):
                training_step += 1

                batch = shuffled_x[i_batch:i_batch+batch_size]
                generated_batch = self.G.predict(self.noise(batch_size))
                x = np.concatenate((batch, generated_batch))
                y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
                self.DM.trainable = True
                d_loss = self.DM.train_on_batch(x, y)
    
                x = self.noise(batch_size)
                y = np.ones([batch_size, 1])
                self.DM.trainable = False
                a_loss = self.AM.train_on_batch(x, y)
                x = self.noise(batch_size)
                a_loss = self.AM.train_on_batch(x, y)
                print("%d: D: loss: %f, acc: %f, A: loss: %f, acc: %f" % (training_step,
                    d_loss[0], d_loss[1],
                    a_loss[0], a_loss[1]
                    ))
            self.plot_images()

    def noise(self, batch_size):
        return np.random.uniform(-1.0, 1.0, size=[batch_size, self.noise_size])

    def plot_images(self, images=None, noise=10, figsize=(15, 1.5), cols=10):
        if images is None:
            if isinstance(noise, int):
                noise = self.noise(noise)
            images = self.G.predict(noise).reshape(-1, self.image_width, self.image_height)

        plt.figure(figsize=figsize)
        rows = np.ceil(len(images) / cols)
        for i, v in enumerate(images):
            plt.subplot(rows, cols, i+1)
            plt.axis('off')
            plt.imshow(v, cmap='gray')
        #plt.savefig(filename)
        #plt.close('all')
        plt.show()

    def save_g_model(self):
        self.G.save('path')

class DCGAN(GAN):

    def __init__(self, image_width, image_height, image_channels, generator=None, discriminator=None, noise_size=100):
        super(DCGAN, self).__init__(image_width, image_height, image_channels, generator, discriminator, noise_size)

    def default_discriminator(): # {{{
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

    def default_generator(): # {{{
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

def test():
    gan = GAN(28, 28, 1)
    check_noise = gan.noise(10)
    gan.fit(x_train) #! callback

    def check_images():
        gan.plot_image(check_noise)

x_train, y_train, x_test, y_test = mnist_data()

if __name__ == '__main__':
    test()
