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
    content_image = tf.image.resize(content_image, [768, 768])

    style_image = tf.io.read_file("./data/style.jpg")
    style_image = tf.image.decode_jpeg(style_image)
    style_image = tf.image.resize(style_image, [768, 768])

    content_image_arr = tf.cast(content_image, tf.float32)
    style_image_arr = tf.cast(style_image, tf.float32)

    content_image_arr = (content_image_arr/127.5)-1
    style_image_arr = (style_image_arr/127.5)-1

    return content_image_arr, style_image_arr


#========================
#MODEL UTILS
#========================
def downsample(filters, size, apply_batchnorm=True):
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
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

    conv1 = tf.keras.layers.Conv2D(512, 4, strides=1,
            kernel_initializer=initializer,
            use_bias=False)(zero_pad1)

    batch_norm1 = tf.keras.layers.BatchNormalization()(conv1)

    leaky_relu = tf.keras.layers.LeakyReLU()(batch_norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
            kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


#========================
#CREATE MODEL OBJECTS
#========================
generator = create_generator()
discriminator = create_discriminator()


#========================
#CREATE OPTIMIZERS
#========================
def create_adam_optim():
    return tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

generator_optimizer = create_adam_optim()
discriminator_optimizer = create_adam_optim()


#========================
#LOSS FUNCTIONS
#========================
LAMBDA = 200
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss(tf.ones_like(disc_generated_output), disc_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  return gan_loss + (LAMBDA * l1_loss)

#========================

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

#========================
#TRAIN STEP
#========================
def train_one_step(content_image, style_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(content_image[tf.newaxis,...], training=True)

        real_outputs = discriminator([content_image[tf.newaxis,...], style_image[tf.newaxis,...]], training=True)
        fake_outputs = discriminator([content_image[tf.newaxis,...], generated], training=True)

        gen_loss = generator_loss(fake_outputs, generated, style_image[tf.newaxis,...])
        disc_loss = discriminator_loss(real_outputs, fake_outputs)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#========================
#TRAIN
#========================
def train(content_image, style_image):
    epochs = 50
    for epoch in range(epochs):
        start = time.time()

        train_one_step(content_image, style_image)

        save_image(generator, content_image, style_image, epoch)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))


#========================
#SAVE IMAGE
#========================
def save_image(model, test_input, tar, epoch):
    prediction = model(test_input[tf.newaxis,...], training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input, tar, prediction[0]]
    title = ['Content Image', 'Style Image', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('./output/epoch-'+str(epoch)+".jpg")


#========================
#METHOD CALLS
#========================
content_image, style_image = load_and_preprocess_data()
train(content_image, style_image)
