import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from keras.layers import add, BatchNormalization, Dropout
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, add
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.layers import Dropout, Input, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from keras import backend as K

tf.keras.backend.set_floatx('float32')

def weighted_mse(y_true, y_pred):
    # Extract the noise mask from the true labels tensor
    y_true, noise_mask = y_true[:, :, :, :1], y_true[:, :, :, 1:]
    
    # Define the weight matrix with higher weight for noise pixels
    weights = K.cast(K.greater(noise_mask, 0.04), 'float32') * 9 + 1  # Increased weight for noise pixels
    
    # Compute the element-wise squared difference
    mse = K.square(y_true - y_pred)
    
    # Compute the weighted mean squared error
    wmse = K.mean(mse * weights)
    
    return wmse


def CDAE(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoding parttf.keras.backend.set_floatx('float32')

    x = Conv2D(32, (5,5), activation='relu', padding='same')(input_img)
    skip = x
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Hidden layer (Bottleneck)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    
    # Decoding part
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = add([x, skip])  # Adding the residual connection
    
    # Output layer with linear activation
    decoded = Conv2D(1, (5, 5), activation='linear', padding='same')(x)
    
    optimizer = RMSprop()
    
    unet = Model(input_img, decoded)
    unet.compile(optimizer=optimizer, loss=weighted_mse)
    
    return unet

def RFDL_RNET7(tensor_shape, activation='relu', trainable=True):
    # Input layer
    
    filters = 32
    kernel_size = 5
    
    # First convolutional layer
    x_in = Input(tensor_shape)
    xp = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', trainable=trainable)(x_in)
    x = BatchNormalization()(xp)
    x = ReLU()(x)

    # Second convolutional layer
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', trainable=trainable)(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Third convolutional layer
    x2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', trainable=trainable)(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Skip connection from the first layer to the third layer
    x3 = Add()([x2, xp])

    # Fourth convolutional layer
    x4 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', trainable=trainable)(x3)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)

    # Fifth convolutional layer
    x6 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same', trainable=trainable)(x4)
    x6 = BatchNormalization()(x6)
    x6 = ReLU()(x6)

    # Skip connection from the third to the fifth layer
    x7 = Add()([x6, x3])

    # Sixth convolutional layer
    x8 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x7)
    x8 = BatchNormalization()(x8)
    x8 = ReLU()(x8)

    # Final output layer
    x_out = Conv2D(filters=1, kernel_size=kernel_size, strides=(1, 1), padding='same', activation=activation)(x8)
    model = Model(inputs=[x_in], outputs=[x_out])

    return model
