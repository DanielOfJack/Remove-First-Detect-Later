import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float32')

def RFINET_downblock(input_tensor, n_filters, kernel_size = 3, batchnorm = True, stride=(1,1)):
    # first layer
    x0 = layers.Conv2D(filters = n_filters, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters = 2*n_filters, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(x0)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    
    x2 = layers.Conv2D(filters = 2*n_filters, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    
    skip = layers.Conv2D(filters = 2*n_filters,
                         kernel_size = (1, 1),\
                         kernel_initializer = 'he_normal',
                         strides=stride,
                         padding = 'same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
                                
    return x

def RFINET_upblock(input_tensor, n_filters, kernel_size = 3, batchnorm = True, stride=(1,1)):
    x0 = layers.Conv2D(filters = n_filters, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters = n_filters//2, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(x0)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    
    x2 = layers.Conv2D(filters = n_filters//2, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      padding = 'same')(x1)
    x2 = layers.BatchNormalization()(x2)
    
    skip = layers.Conv2D(filters = n_filters//2,
                         kernel_size = (1, 1),\
                         kernel_initializer = 'he_normal',
                         strides=stride,
                         padding = 'same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
                                
    return x

def RFINET(input_shape, activation, n_filters = 32, dropout = 0.05, batchnorm = True):
    
    input_data = tf.keras.Input(input_shape,name='data') 
    c0 = layers.Conv2D(filters = 32, 
                      kernel_size = (3, 3),\
                      kernel_initializer = 'he_normal', 
                      strides=1,
                      padding = 'same')(input_data)

    c1 = RFINET_downblock(c0,n_filters * 1, kernel_size = 3, batchnorm = batchnorm, stride=(1,1))
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(p1)

    c2 = RFINET_downblock(p1, n_filters * 2, kernel_size = 3, stride=(1,1), batchnorm = batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(p2)
    
    c3 = RFINET_downblock(p2, n_filters * 4, kernel_size = 3, stride=(1,1), batchnorm = batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout)(p3)
    
    c4 = RFINET_downblock(p3, n_filters * 8, kernel_size = 3, stride=(1,1),batchnorm = batchnorm)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(dropout)(p4)

    c5 = RFINET_downblock(p4, n_filters * 16, kernel_size = 3, stride=(1,1), batchnorm = batchnorm)
    p5 = layers.MaxPooling2D((2, 2))(c5)
    p5 = layers.Dropout(dropout)(p5)
    
    # upsampling 
    u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = RFINET_upblock(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = RFINET_upblock(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = RFINET_upblock(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = RFINET_upblock(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model
