import numpy as np
import argparse
import tensorflow as tf
import os
import pickle
import pandas as pd
import gc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from utils.data import split_into_patches, normalize_patches
from utils.callbacks import get_callbacks
from architectures.RNET import RNET7, RNET6, RNET5
from architectures.UNET import UNET
from architectures.RFINET import RFINET
from architectures.RFDL import RFDL_RNET7

class Convolutional:
    def __init__(self, **kwargs):
            self.save_name = kwargs.get('save_name', 'MODEL')
            self.model_name = kwargs.get('model_name', 'RNET7')
            self.dataset = kwargs.get('dataset', 'LOFAR')
            self.patch_sz = kwargs.get('patch_sz', 512)
            self.loss = kwargs.get('loss', 'mse')
            self.learning_rate = kwargs.get('learning_rate', 0.001)
            self.learning_rate_scheduler = kwargs.get('learning_rate_scheduler')
            self.fold = kwargs.get('fold', 0)
            self.epochs = kwargs.get('epochs', 50)
            self.activation, self.optimizer = self.get_compilation()
            self.batch_size = self.get_batch_size()
            self.log_name = self.get_log_name()

    def get_batch_size(self):
        if self.patch_sz == 64:
            return 64
        else:
            return 16

    def get_compilation(self):
        if self.loss == 'binary_crossentropy':
            return "sigmoid", tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        else:
            return "relu", tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

    def get_log_name(self):
        if self.fold > 0:
            return f"{self.save_name}_{self.fold}.csv"
        else:
            return self.save_name

    def compile_model(self):
        tensor_shape = (self.patch_sz, self.patch_sz, 1)
        
        model_map = {
            "UNET": UNET,
            "RNET7": RNET7,
            "RNET6": RNET6,
            "RNET5": RNET5,
            "RFINET":RFINET,
            "RFDL": RFDL_RNET7,
        }
        
        model = model_map[self.model_name](tensor_shape, self.activation)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model
    
    def compile_transfer_model(self):
        tensor_shape = (self.patch_sz, self.patch_sz, 1)
        
        model_map = {
            "UNET": UNET,
            "RNET7": RNET7,
            "RNET6": RNET6,
            "RNET5": RNET5,
            "RFINET":RFINET,
            "RFDL": RFDL_RNET7,
        }
        
        loaded_model = load_model(f"models/HERA/{self.model_name}_A.h5")
        
        model = model_map[self.model_name](tensor_shape, self.activation, trainable=False)
        model.set_weights(loaded_model.get_weights())
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val):
        callbacks = get_callbacks(self)
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks)
        score = callbacks[0].best
        return model, score
