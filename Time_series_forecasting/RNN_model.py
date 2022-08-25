import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean
from tensorflow.keras.models import Model
from tensorflow.python.keras.initializers import RandomUniform
import numpy as np


class RNN_model:
    def __init__(self, input_shape, output_shape, lr=0.01, units=None):
        if units is None:
            units = [512, 32] # GRU layers units
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lr = lr
        self.model = None
        self.units = units

    def define_RNN_model(self):
        input_layer = Input(shape=self.input_shape)
        GRU_layer = GRU(units=self.units[0], return_sequences=True)(input_layer)
        GRU_layer2 = GRU(units=self.units[1], return_sequences=False)(GRU_layer)
        output_layer = Dense(5, name="output")(GRU_layer2)
        init = RandomUniform(minval=-0.05, maxval=0.05)
        x = Dense(self.output_shape, activation='linear', kernel_initializer=init)(output_layer)
        self.model = Model(input_layer, x, name='RNN')
        optimizer = RMSprop(lr=self.lr)
        self.model.compile(loss='mse', optimizer=optimizer)
        self.model.summary()

    def load_model(self, name):
        self.model = tf.keras.models.load_model("models/{}.h5".format(name), compile=True)

    def define_callbacks(self):
        """ Callbacks to be called under fitting"""
        callback_checkpoint = ModelCheckpoint(filepath='24_12_checkpoint', monitor='val_loss', verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)
        return [callback_early_stopping, callback_checkpoint, callback_reduce_lr]

    def plot_loss(self, history):
        """ Plot loss and validation loss """
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def fit_model_(self, generator, validation_data, callbacks):
        """ Fit the model using a generator with some random sequences from the data """
        history = self.model.fit(x=generator, epochs=20, steps_per_epoch=200, validation_data=validation_data,
                                 callbacks=callbacks)
        self.model.save('models/{}.h5'.format('24_10_alt'))
        self.plot_loss(history)
        return history

    def fit_model(self, generator, validation_data, callbacks):
        """ Fit the model using all sequences in the data set """
        history = self.model.fit(x=generator[0], y=generator[1], epochs=6, validation_data=validation_data,
                                 callbacks=callbacks)
        self.model.save('models/{}.h5'.format('24_12'))
        self.plot_loss(history)
        return history

    def load_checkpoint(self, path_checkpoint):
        """ If the weights are saved, load them """
        if self.model is None:
            print('There is no model!')
        try:
            self.model.load_weights(path_checkpoint)
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

    def predict(self, x):
        """ Predict the next target for one sequence """
        return self.model.predict(x, verbose=False)

    def evaluate_test(self, x_test, y_test):
        result = self.model.evaluate(x=x_test, y=y_test)
        return result
