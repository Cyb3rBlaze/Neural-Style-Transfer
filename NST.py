import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

#========================
#LOAD AND PREPROCESS DATA
#========================
def load_and_preprocess_data():
    content_image = tf.io.read_file("./data/content.jpg")
    content_image = tf.image.decode_jpeg(content_image)

    style_image = tf.io.read_file("./data/style.jpg")
    style_image = tf.image.decode_jpeg(style_image)

    content_image_arr = tf.cast(content_image, tf.float32)
    style_image_arr = tf.cast(style_image, tf.float32)

    return content_image_arr, np.array(style_image_arr)


#========================
#MODEL UTILS
#========================
def downsample(filters, size, apply_batchnorm=True):
    #Reduces variance of the image to prevent exploding gradients
    initializer = tf.random_normal_initializer(0., 0.02)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    return model

#========================

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same",
        kernel_initializer=initializer, use_bias=False))

    model.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.ReLU())

    return model


#========================
#CREATE GENERATOR
#========================
def create_generator():
    #1024x768x3
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),#512x384x64
        downsample(128, 4),#256x192x128
        downsample(256, 4),#128x96x256
        downsample(512, 4),#64x48x512
        downsample(512, 4),#32x24x512
        downsample(512, 4),#16x12x512
        downsample(512, 4),#8x6x512
        downsample(512, 4),#4x3x512
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),#8x6x1024
        upsample(512, 4, apply_dropout=True),#16x12x1024
        upsample(512, 4, apply_dropout=True),#32x24x1024
        upsample(512, 4),#64x48x1024
        upsample(256, 4),#128x96x512
        upsample(128, 4),#256x192x256
        upsample(64, 4)#512x384x128
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
            strides=2,
            padding='same',
            kernel_initializer=initializer,
            activation='tanh')

    concatenate = tf.keras.layers.Concatenate()

    input = tf.keras.layers.Input(shape=[None, None, 3])
    x = input

    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concatenate([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=input, outputs=x)


#========================
#CREATE DISCRIMINATOR
#========================
def create_discriminator():
    pass


#========================
#CREATE MODEL OBJECTS
#========================
generator = create_generator()
#discriminator = create_discriminator()


#========================
#CREATE OPTIMIZERS
#========================
def create_adam_optim():
    return tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator_optimizer = create_adam_optim()
discriminator_optimizer = create_adam_optim()


#========================
#GENERATOR LOSS
#========================
def generator_loss():
    pass


#========================
#DISCRIMINATOR LOSS
#========================
def discriminator_loss():
    pass


#========================
#TRAIN STEP
#========================
def train_one_step():
    pass


#========================
#TRAIN
#========================
def train(content_image, style_image):
    pass


#========================
#SAVE IMAGE
#========================
def save_image():
    pass


#========================
#METHOD CALLS
#========================
content_image, style_image = load_and_preprocess_data()
new_image = np.array(generator(content_image[tf.newaxis,...], training=False))
fig, ax = plt.subplots(1, 1)
ax.imshow(new_image[0,...], interpolation='none')
ax.axis("off")
fig.savefig('./output/test.jpg')
#train(content_image, style_image)
