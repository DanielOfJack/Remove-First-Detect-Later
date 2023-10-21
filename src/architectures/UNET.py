import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float32')

def Conv2D_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, stride=(1,1), trainable=True):
    # first layer
    x = layers.Conv2D(filters = n_filters, 
                      kernel_size = (kernel_size, kernel_size),\
                      kernel_initializer = 'he_normal', 
                      strides=stride,
                      trainable=trainable,
                      padding = 'same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def UNET(tensorshape, activation, optimizer="RMSprop", ksz=3, n_filters=16, loss="binary_crossentropy", met=0, dropout = 0.05, batchnorm = True, trainable=True):
    # Contracting Path
    input_data = tf.keras.Input(tensorshape, name='data')
    _str = 2
    c1 = Conv2D_block(input_data, 
                      n_filters * 1, 
                      kernel_size = ksz, 
                      batchnorm = batchnorm, 
                      trainable=trainable,
                      stride=(_str,_str))
    #p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(c1)
    
    c2 = Conv2D_block(p1, 
                      n_filters * 2, 
                      kernel_size = ksz, 
                      stride=(2,2),
                      batchnorm = batchnorm, 
                      trainable=trainable)
    #p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(c2)
    
    c3 = Conv2D_block(p2, 
                      n_filters * 4, 
                      kernel_size = ksz, 
                      stride=(2,2),
                      batchnorm = batchnorm,
                      trainable=trainable)
    #p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout)(c3)

    c4 = Conv2D_block(p3, 
                      n_filters * 8, 
                      kernel_size = ksz, 
                      stride=(2,2),
                      batchnorm = batchnorm,
                      trainable=trainable)
    #p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(dropout)(c4)
    
    c5 = Conv2D_block(p4, n_filters = n_filters * 16, kernel_size = ksz, stride=(2,2), batchnorm = batchnorm)
    
    # Expansive Path
    u6 = layers.Conv2DTranspose(n_filters * 8, (ksz, ksz), strides = (2, 2), padding = 'same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(dropout)(u6)
    c6 = Conv2D_block(u6, n_filters * 8, kernel_size = ksz, batchnorm = batchnorm)
    
    u7 = layers.Conv2DTranspose(n_filters * 4, (ksz, ksz), strides = (2, 2), padding = 'same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(dropout)(u7)
    c7 = Conv2D_block(u7, n_filters * 4, kernel_size = ksz, batchnorm = batchnorm)

    u8 = layers.Conv2DTranspose(n_filters * 2, (ksz, ksz), strides = (2, 2), padding = 'same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = Conv2D_block(u8, n_filters * 2, kernel_size = ksz, batchnorm = batchnorm)
    
    u9 = layers.Conv2DTranspose(n_filters * 1, (ksz, ksz), strides = (2, 2), padding = 'same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    u9 = layers.UpSampling2D((2,2))(u9)
    c9 = Conv2D_block(u9, n_filters * 1, kernel_size = ksz, batchnorm = batchnorm)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    
    if met == 1:
        metrics = ['accuracy']
    else:
        metrics = metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

