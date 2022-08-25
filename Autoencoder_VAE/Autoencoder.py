import tensorflow as tf
import tensorflow_probability as tf_p
from tensorflow import keras
import numpy as np
from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet
import matplotlib.pyplot as plt


class Encoder:
    ''' The encoder part '''
    def __init__(self, shape, vae, bottle_neck_size):
        self.input_layer = keras.Input(shape=shape)
        self.input_layer_flatten = keras.layers.Flatten()(self.input_layer)
        self.next_layer = keras.layers.Dense(400, activation="sigmoid")(self.input_layer_flatten)
        self.next_layer2 = keras.layers.Dense(120, activation="sigmoid")(self.next_layer)
        self.next_layer3 = keras.layers.Dense(60, activation="relu")(self.next_layer2)
        if vae:
            # If we have VAE I take the mean and log var as meu and sigma then I call sample with the previose layer.
            self.meu = keras.layers.Dense(bottle_neck_size)(self.next_layer3)
            self.sigma = keras.layers.Dense(bottle_neck_size)(self.next_layer3)
            self.output_layer = keras.layers.Lambda(self.sample_output_layer_vae, output_shape=(bottle_neck_size,))(
                [self.meu, self.sigma])
            self.encoder_model = keras.Model(self.input_layer, [self.meu, self.sigma, self.output_layer], name='encoder')
        else:
            self.output_layer = keras.layers.Dense(bottle_neck_size, activation="linear")(self.next_layer3)
            self.encoder_model = keras.Model(self.input_layer, self.output_layer, name='Encoder')

    def sample_output_layer_vae(self, meu_sigma):
        # the sample function to find meu + sigma * epsilon
        meu, sigma = meu_sigma
        epsilon_shape = keras.backend.shape(meu)
        epsilon = keras.backend.random_normal(shape=epsilon_shape)
        return meu + keras.backend.exp(sigma / 2) * epsilon


class Decoder:
    ''' The decoder part '''
    def __init__(self, bottle_neck, shape, vae, encoder=None):
        self.input_layer = keras.layers.Dense(60, activation="relu")(bottle_neck)
        self.next_layer = keras.layers.Dense(120, activation="sigmoid")(self.input_layer)
        self.next_layer2 = keras.layers.Dense(400, activation="sigmoid")(self.next_layer)
        self.output_not_shaped = keras.layers.Dense(28 * 28, activation="sigmoid")(self.next_layer2)
        self.output_layer = keras.layers.Reshape(shape)(self.output_not_shaped)
        if vae:
            # If we have VAE I add a new probabilistic layer with meu and sigma from the decoder part to make the distribution
            # then I make it as a last layer of the decoder.
            probabilistic_layer_decoder = ProbabilisticLayer()([encoder.input_layer, self.output_layer],
                                                               encoder.meu, encoder.sigma)
            self.output_layer = probabilistic_layer_decoder
        # self.decoder_model = keras.Model(self.input_layer, self.output_layer, name='Decoder')


class ProbabilisticLayer(keras.layers.Layer):
    # This layer is an updated version of keras.layers.Layer to update the loss in VAE with the probabilistic distribution.
    # then it return the predicted images.
    def call(self, inputs, meu, sigma):
        flatten_images, flatten_predicted_images = keras.backend.flatten(inputs[0]), keras.backend.flatten(inputs[1])
        predict_loss = keras.metrics.binary_crossentropy(flatten_images, flatten_predicted_images)
        KL = keras.backend.mean(1 + sigma - keras.backend.square(meu) - keras.backend.exp(sigma), axis=-1) * (-1e-4)
        loss = keras.backend.mean(predict_loss + KL)
        self.add_loss(loss, inputs=inputs)
        return inputs[1]


class Autoencoder:
    ''' The Autoencoder, takes the mode of images as input and if it's AE or VAE, and the middle layer size'''
    # It works on one channel so I make all the channels in one and send it to the autoencoder
    def __init__(self, mono=True, binary=False, complete=False, vae=False, bottle_neck_size=5, anom=False):
        self.mono = mono
        self.binary = binary
        self.complete = complete
        self.vae = vae
        self.anom = anom
        self.train_images = None
        self.test_images = None
        self.test_labels = None
        self.train_labels = None
        self.random_images = None

        self.name = 'VAE' if self.vae else 'AE'

        self.gen = self.getData()
        self.encoder = Encoder(shape=self.train_images[0].shape, vae=self.vae, bottle_neck_size=bottle_neck_size)
        if self.vae:
            self.decoder = Decoder(self.encoder.output_layer, self.train_images[0].shape, self.vae, encoder=self.encoder)
        else:
            self.decoder = Decoder(self.encoder.output_layer, self.train_images[0].shape, self.vae)

        if vae:
            opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8)
            self.autoencoder_model = keras.Model(self.encoder.input_layer, self.decoder.output_layer, name='VAE')
            self.autoencoder_model.summary()
            # the loss is None here because we are using the probabilistic layer to update the loss.
            self.autoencoder_model.compile(optimizer=opt, loss=None)

        else:
            # It's better to use binary_crossentropy as loss function when the image is binary
            if binary:
                loss = 'binary_crossentropy'
            else:
                loss = 'mse'
            opt = tf.keras.optimizers.Adam(learning_rate=0.0009, decay=1e-7)
            self.autoencoder_model = keras.Model(self.encoder.input_layer, self.decoder.output_layer, name='AE')
            self.autoencoder_model.summary()
            self.autoencoder_model.compile(opt, loss=loss)

    def split_3images(self, images):
        # if we have 3 channels this function will be used to add all of the images into one array
        # to make it one channel
        all_images_splited = []
        for image in images:
            for i in range(3):
                all_images_splited.append(np.expand_dims(image[:, :, i], axis=2))
        return np.array(all_images_splited)

    def split_3lables(self, lables):
        # make all labels in the 3 channels as one array with each value as label
        all_lables_splited = []
        for lable in lables:
            str_lable = str(lable)
            if len(str_lable) == 1:
                str_lable = '00' + str_lable
            elif len(str_lable) == 2:
                str_lable = '0' + str_lable
            for char in reversed(str_lable):
                all_lables_splited.append(int(char))
        return np.array(all_lables_splited)

    def collect_images(self, images, lables, gen, name=''):
        # After training and prediction, the one channel big array of images will be collected back into 3 channels
        all_collected_images = []
        all_collected_lables = []
        for i in range(0, len(images), 3):
            one_collected_image = np.stack((images[i].reshape(28, 28), images[i + 1].reshape(28, 28),
                                      images[i + 2].reshape(28, 28)), axis=2)
            all_collected_images.append(one_collected_image)
            one_collected_lable = int(str(int(lables[i]))+str(int(lables[i+1]))+str(int(lables[i+2])))
            all_collected_lables.append(one_collected_lable)
        for batch in range(0, 18, 9):
            gen.plot_example(images=np.array(all_collected_images[batch:batch+9]),
                             labels=np.array(all_collected_lables[batch:batch+9]), name='{}{}{}'.format(name, self.name, batch))
        # gen.plot_example(images=np.array([one_collected_image]), labels=np.array([one_collected_lable]),
        # name='{}{}'.format(name, self.name))
        return np.array(all_collected_images), all_collected_lables

    def check_datamode(self, complete=None):
        if self.mono and self.binary and self.complete:
            data_mode = DataMode.MONO_BINARY_COMPLETE
        elif self.mono and self.binary and (not self.complete):
            if complete is not None:
                data_mode = DataMode.MONO_BINARY_COMPLETE
            else:
                data_mode = DataMode.MONO_BINARY_MISSING
        elif self.mono and (not self.binary) and self.complete:
            data_mode = DataMode.MONO_FLOAT_COMPLETE
        elif self.mono and (not self.binary) and (not self.complete):
            if complete is not None:
                data_mode = DataMode.MONO_FLOAT_COMPLETE
            else:
                data_mode = DataMode.MONO_FLOAT_MISSING
        elif (not self.mono) and self.binary and self.complete:
            data_mode = DataMode.COLOR_BINARY_COMPLETE
        elif (not self.mono) and self.binary and (not self.complete):
            if complete is not None:
                data_mode = DataMode.COLOR_BINARY_COMPLETE
            else:
                data_mode = DataMode.COLOR_BINARY_MISSING
        elif (not self.mono) and (not self.binary) and self.complete:
            data_mode = DataMode.COLOR_FLOAT_COMPLETE
        elif (not self.mono) and (not self.binary) and (not self.complete):
            if complete is not None:
                data_mode = DataMode.COLOR_FLOAT_COMPLETE
            else:
                data_mode = DataMode.COLOR_FLOAT_MISSING
        else:
            data_mode = None
            print('something wrong with generating!')
        if not data_mode:
            return False
        return data_mode


    def getData(self):
        """ Generate the images with the specified mode and put them in one array if they are in 3 channels"""
        data_mode = self.check_datamode()
        gen = StackedMNISTData(mode=data_mode, default_batch_size=1)
        if not self.mono:
            self.train_images = self.split_3images(gen.train_images)
            self.test_images = self.split_3images(gen.test_images)
            self.train_labels = self.split_3lables(gen.train_labels)
            self.test_labels = self.split_3lables(gen.test_labels)
            # self.collect_images(self.train_images, self.train_labels, gen, name='collectedOriginal')
        else:
            self.train_images = gen.train_images
            self.test_images = gen.test_images
            self.train_labels = gen.train_labels
            self.test_labels = gen.test_labels

        # img, cls = gen.get_random_batch(batch_size=1)
        # gen.plot_example(images=img, labels=cls)

        return gen

    def train(self, epochs):
        # train the auto encoder and show the learned weights
        for _ in range(epochs):
            self.autoencoder_model.fit(self.train_images, self.train_images, epochs=1, batch_size=10)
        for arr in self.autoencoder_model.trainable_variables:
            if len(arr.shape) > 1:
                plt.imshow(arr)
                plt.show()

    def predict_some(self, images, lables):
        # predict the test imaged and find the reconstructed loss and save images in images folder
        predicted_images = self.autoencoder_model.predict(images)
        reconstruct_loss = keras.metrics.binary_crossentropy(images, predicted_images)
        print('Reconstruct Loss', reconstruct_loss)
        x = 1000
        for index, img in enumerate(predicted_images):
            if index % x == 0:
                plt.imshow(images[index])
                plt.show()
                plt.imshow(img)
                plt.show()
                if self.vae:
                    plt.imsave('images/predicted{i}VAE.png'.format(i=index), img.reshape(28, 28))
                    plt.imsave('images/original{i}VAE.png'.format(i=index), images[index].reshape(28, 28))
                else:
                    plt.imsave('images/predicted{i}AE.png'.format(i=index), img.reshape(28, 28))
                    plt.imsave('images/original{i}AE.png'.format(i=index), images[index].reshape(28, 28))
        return predicted_images

    def predict_anom(self):
        # Generate the complete image set and use it to find the reconstructed loss from the trained autoencoder
        data_mode = self.check_datamode(complete=True)
        gen = StackedMNISTData(mode=data_mode, default_batch_size=1)
        if not self.mono:
            anom_test_images = self.split_3images(gen.test_images)
            anom_test_labels = self.split_3lables(gen.test_labels)
        else:
            anom_test_images = gen.test_images
            anom_test_labels = gen.test_labels
        predicted_images = self.autoencoder_model.predict(anom_test_images)
        reconstruct_loss = keras.metrics.binary_crossentropy(anom_test_images, predicted_images)
        print('Reconstruct Loss Anom', reconstruct_loss)
        if not self.mono:
            # collect the images to 3 channels again
            self.collect_images(anom_test_images, anom_test_labels, gen, name='anom')
            self.collect_images(predicted_images, anom_test_labels, gen, name= 'anom')
        else:
            x = 1000
            for index, img in enumerate(predicted_images):
                if index % x == 0:
                    plt.imshow(anom_test_images[index])
                    plt.show()
                    plt.imshow(img)
                    plt.show()
                    if self.vae:
                        plt.imsave('images/anom/predicted_anom{i}VAE.png'.format(i=index), img.reshape(28, 28))
                        plt.imsave('images/anom/original_anom{i}VAE.png'.format(i=index),
                                   anom_test_images[index].reshape(28, 28))
                    else:
                        plt.imsave('images/anom/predicted_anom{i}AE.png'.format(i=index), img.reshape(28, 28))
                        plt.imsave('images/anom/original_anom{i}AE.png'.format(i=index), anom_test_images[index].reshape(28, 28))

    def generate_random_images(self, images_size, VN):
        """ Auto encoder as a generator """
        # produce random matrices and apply predict on them
        self.random_images = np.array([np.expand_dims(np.random.rand(28, 28), axis=2) for _ in range(images_size * self.gen.channels)])
        # self.random_images = np.array([np.random.rand(28, 28, self.gen.channels) for _ in range(images_size)])
        predicted_images = self.autoencoder_model.predict(self.random_images)
        lables, beliefs = VN.predict(predicted_images)
        if self.gen.channels > 1:
            predicted_images, lables = self.collect_images(predicted_images, lables, self.gen, name='gen/collected{}'.format(self.name))
        x = 1
        for index, img in enumerate(predicted_images):
            if index % x == 0:
                plt.title('Random image')
                plt.imshow(self.random_images[index])
                plt.show()
                plt.title(lables[index])
                plt.imshow(img)
                plt.show()
                '''if self.vae:
                    plt.imsave('images/gen/predicted_gen{i}VAE.png'.format(i=index), img.reshape(28, 28))
                    plt.imsave('images/gen/original_gen{i}VAE.png'.format(i=index),
                               self.random_images[index].reshape(28, 28))
                else:
                    plt.imsave('images/gen/predicted_gen{i}AE.png'.format(i=index), img.reshape(28, 28))
                    plt.imsave('images/gen/original_gen{i}AE.png'.format(i=index),
                               self.random_images[index].reshape(28, 28))'''
        return self.random_images

    def make_separate_channle_images(self, images, lables, color_number, steps):
        # split images to different channels from the big produced array from split3_images function
        channel_images = []
        channel_lables = []
        for i in range(color_number, len(images), steps):
            channel_images.append(images[i])
            channel_lables.append(lables[i])
        return np.array(channel_images), np.array(channel_lables)

    def validation_net_each_channel(self, verification_net, test_images, test_labels, channel_number):
        # validate based on channel number and return the channel accuray and predictibility
        steps = 1 if self.mono else 3
        channel_test_images, channel_test_labels = self.make_separate_channle_images(test_images, test_labels,
                                                                                     channel_number, steps)
        predicted_images = self.predict_some(channel_test_images, channel_test_labels)

        cov = verification_net.check_class_coverage(data=predicted_images, tolerance=.8)
        pred, acc = verification_net.check_predictability(data=predicted_images, correct_labels=np.array(channel_test_labels))
        return cov, pred, acc, predicted_images

    def validation_net_all_channels(self, verification_net, test_images, test_labels):
        # validate based on 3 channels with 3 labels in each image
        steps = 1 if self.mono else 3
        all_channels_labels = []
        for i in range(0, len(test_labels), steps):
            one_labels = []
            for step in range(steps):
                one_labels.append(test_labels[i + step])
            all_channels_labels.append(one_labels)

        predicted_images = self.predict_some(test_images, test_labels)

        cov = verification_net.check_class_coverage(data=predicted_images, tolerance=.5)
        pred, acc = verification_net.check_predictability_3C(data=predicted_images,
                                                             correct_labels=np.array(all_channels_labels), tolerance=.5)
        print(f"Coverage 3channels: {100 * cov:.2f}%")
        print(f"Predictability 3channels: {100 * pred:.2f}%")
        print(f"Accuracy 3channels: {100 * acc:.2f}%")

    def validate(self):
        # Call the classifier verification_net and train it, then validate with out test images
        data_mode = self.check_datamode()
        gen = StackedMNISTData(mode=data_mode, default_batch_size=2048)
        verification_net = VerificationNet()

        if not self.mono:
            train_images = self.split_3images(gen.train_images)
            test_images = self.split_3images(gen.test_images)
            train_labels = self.split_3lables(gen.train_labels)
            test_labels = self.split_3lables(gen.test_labels)
        else:
            train_images = gen.train_images
            test_images = gen.test_images
            train_labels = gen.train_labels
            test_labels = gen.test_labels

        verification_net.train(train_images, test_images, train_labels, test_labels, epochs=5)

        cov = np.zeros(self.gen.channels)
        pred = np.zeros(self.gen.channels)
        acc = np.zeros(self.gen.channels)

        if not self.mono:
            self.validation_net_all_channels(verification_net, self.test_images, self.test_labels)

        # Here I validate each channel alone then I find the overall accuracy for all channels
        predicted_images_all_channels = {}
        all_predicted_images = []
        for channel in range(self.gen.channels):
            cov[channel], pred[channel], acc[channel], predicted_images_all_channels[channel] = \
                self.validation_net_each_channel(verification_net, self.test_images, self.test_labels, channel)

        for i in range(len(predicted_images_all_channels[0])-1):
            for channel in range(self.gen.channels):
                all_predicted_images.append(predicted_images_all_channels[channel][i])
        if not self.mono:
            self.collect_images(self.test_images, self.test_labels, gen, name='collectedOriginalValidated')
            self.collect_images(all_predicted_images, self.test_labels, gen, name='collected_predictedValidated')

        print(f"Coverage: {100 * cov.mean():.2f}%")
        print(f"Predictability: {100 * pred.mean():.2f}%")
        print(f"Accuracy: {100 * acc.mean():.2f}%")
        return verification_net

def main():
    mono = bool(int(input('mono=1, color=0:  ')))
    binary = bool(int(input('binary=1, float=0:  ')))
    complete = bool(int(input('complete=1, missing=0:  ')))
    vae = bool(int(input('VAE=1, AE=0:  ')))
    anom = bool(int(input('anom=1, else=0:  ')))
    middel_layer_size = int(input('middel layer size:  '))
    AE = Autoencoder(mono=mono, binary=binary, complete=complete, vae=vae, bottle_neck_size=middel_layer_size, anom=anom)
    # AE = Autoencoder(mono=True, binary=False, complete=False, vae=False, bottle_neck_size=10, anom=True)
    AE.train(7)
    VN = AE.validate()
    AE.generate_random_images(20, VN)
    AE.predict_anom()


main()