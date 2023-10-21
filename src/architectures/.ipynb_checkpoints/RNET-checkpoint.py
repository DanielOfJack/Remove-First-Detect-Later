from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.layers import MultiHeadAttention


def RNET3(tensor_shape, activation):

    x_in = Input(tensor_shape)
    # First convolutional layer
    xp = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', activation='relu', kernel_initializer=initializer)(x_in)

    # Second convolutional layer
    x1 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', activation='relu', kernel_initializer=initializer)(xp)

    # Output layer
    x_out = Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=activation, kernel_initializer=initializer)(x1)
    
    model = Model(inputs=[x_in], outputs=[x_out])

    return model

def RNET5(tensor_shape, activation, trainable=True):

    x_in = Input(tensor_shape)
    # First convolutional layer
    xp = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', activation='relu', trainable=trainable)(x_in)

    # Second convolutional layer
    x1 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', activation='relu', trainable=trainable)(xp)

    # Third convolutional layer
    x2 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', activation='relu', trainable=trainable)(x1)

    # Skip connection from the first layer to the third layer
    x3 = Add()([x2, xp])
    x3 = BatchNormalization()(x3)

    # Fourth convolutional layer
    x4 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', activation='relu')(x3)

    # Final output layer
    x_out = Conv2D(filters=1, kernel_size=5, strides=(1, 1), 
                   padding='same', activation=activation)(x4)

    model = Model(inputs=[x_in], outputs=[x_out])

    return model

def RNET6(tensor_shape, activation, trainable=True):
    
    x_in = Input(tensor_shape)
    # First convolutional block
    xp = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', trainable=trainable)(x_in)
    x = BatchNormalization()(xp)
    x = ReLU()(x)

    # Second convolutional block
    x1 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', trainable=trainable)(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Third convolutional block
    x2 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), 
                padding='same', trainable=trainable)(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Skip connection from the first block to the third block
    x3 = Add()([x2, xp])

    # Fourth convolutional block
    x4 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)

    # Fifth convolutional block
    x5 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = ReLU()(x5)

    # Final output layer
    x_out = Conv2D(filters=1, kernel_size=5, strides=(1, 1), 
                   padding='same', activation=activation)(x5)

    model = Model(inputs=[x_in], outputs=[x_out])

    return model


def RNET7(tensor_shape, activation='relu', trainable=True):
    # Input layer
    
    # First convolutional layer
    x_in = Input(tensor_shape)
    xp = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', trainable=trainable)(x_in)
    x = BatchNormalization()(xp)
    x = ReLU()(x)

    # Second convolutional layer
    x1 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', trainable=trainable)(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    # Third convolutional layer
    x2 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', trainable=trainable)(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    # Skip connection from the first layer to the third layer
    x3 = Add()([x2, xp])

    # Fourth convolutional layer
    x4 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', trainable=trainable)(x3)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)

    # Fifth convolutional layer
    x6 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same', trainable=trainable)(x4)
    x6 = BatchNormalization()(x6)
    x6 = ReLU()(x6)

    # Skip connection from the third to the fifth layer
    x7 = Add()([x6, x3])

    # Sixth convolutional layer
    x8 = Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x7)
    x8 = BatchNormalization()(x8)
    x8 = ReLU()(x8)

    # Final output layer
    x_out = Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=activation)(x8)
    model = Model(inputs=[x_in], outputs=[x_out])

    return model
