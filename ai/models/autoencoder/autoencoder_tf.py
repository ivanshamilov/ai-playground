from keras.layers import Input, Flatten, Dense, Conv2D, Conv2DTranspose, Dropout, BatchNormalization, LeakyReLU, Reshape, Activation
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf

from PIL import Image

from typing import Tuple, List


class AutoEncoder():

  def __init__(self, 
               input_dim: Tuple[int],
               encoder_conv_filters: List[int],
               encoder_conv_kernel_size: List[int],
               encoder_conv_strides: List[int],
               decoder_conv_t_filters: List[int],
               decoder_conv_t_kernel_size: List[int],
               decoder_conv_t_strides: List[int],
               z_dim: int,
               use_batch_norm=False,
               use_dropout=False):

    self.name = "autoencoder"
    self.input_dim = input_dim
    self.encoder_conv_filters = encoder_conv_filters
    self.encoder_conv_kernel_size = encoder_conv_kernel_size
    self.encoder_conv_strides = encoder_conv_strides
    self.decoder_conv_t_filters = decoder_conv_t_filters
    self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
    self.decoder_conv_t_strides = decoder_conv_t_strides
    self.z_dim = z_dim
    self.use_batch_norm = use_batch_norm
    self.use_dropout = use_dropout
    self.n_layers_encoder = len(encoder_conv_filters)
    self.n_layers_decoder = len(decoder_conv_t_filters)

    self._build()
  
  def _build(self):
    self._build_encoder()
    self._build_decoder()
    model_input = self.encoder_input
    model_output = self.decoder(self.encoder_output)

    self.model = Model(model_input, model_output)

  def _build_encoder(self):
    self.encoder_input = x = Input(shape=self.input_dim, name="encoder_input")

    for i in range(self.n_layers_encoder):
      conv_layer = Conv2D(filters=self.encoder_conv_filters[i],
                          kernel_size=self.encoder_conv_kernel_size[i],
                          strides=self.encoder_conv_strides[i],
                          padding="same",
                          name=f"encoder_conv_{i}")

      x = conv_layer(x)
      x = LeakyReLU(name=f"leaky_relu_{i}")(x)

      if self.use_batch_norm:
        x = BatchNormalization()(x)

      if self.use_dropout:
        x = Dropout(p=0.25)(x)

    self.shape_before_flatten = K.int_shape(x)[1:]
    x = Flatten()(x)

    self.encoder_output = Dense(units=self.z_dim, name="encoder_output")(x)
    self.encoder = Model(self.encoder_input, self.encoder_output)

  def _build_decoder(self):
    decoder_input = x = Input(shape=(self.z_dim, ), name="decoder_input")
    x = Dense(units=np.prod(self.shape_before_flatten), name="dense_1")(x)
    x = Reshape(self.shape_before_flatten)(x)

    for i in range(self.n_layers_decoder):
      conv_layer = Conv2DTranspose(filters=self.decoder_conv_t_filters[i],
                          kernel_size=self.decoder_conv_t_kernel_size[i],
                          strides=self.decoder_conv_t_strides[i],
                          name=f"decoder_conv_t_{i}")
      x = conv_layer(x)
      if i < self.n_layers_decoder - 1:
        x = LeakyReLU()(x)

    decoder_output = Activation("relu", name="relu_activation")(x)
    self.decoder = Model(decoder_input, decoder_output, name="decoder_network")

  def loss_function(y_true, 
                    y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

  def compile(self, optimizer):
    self.model.compile(optimizer=optimizer, loss=self.loss_function)

  def fit(self, 
          x_train, 
          y_train, 
          batch_size, 
          epochs, 
          callbacks_list):
    history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)
    return history 


if __name__ == "__main__":
  ae = AutoEncoder(input_dim=(28, 28, 1),
                   encoder_conv_filters=[32, 64, 64, 64],
                   encoder_conv_kernel_size=[3, 3, 3, 3],
                   encoder_conv_strides=[1, 2, 2, 1],
                   decoder_conv_t_filters=[64, 64, 32, 1],
                   decoder_conv_t_kernel_size=[3, 3, 3, 3],
                   decoder_conv_t_strides=[1, 2, 2, 1],
                   z_dim=2)
                
  ae.compile(optimizer=Adam(learning_rate=0.001))
  # ae.fit(X_train, y_train, batch_size=32, epochs=10)
