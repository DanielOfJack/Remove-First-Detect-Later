import numpy as np
import argparse
import tensorflow as tf
import os
import pickle
import pandas as pd
import gc
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, Callback, LearningRateScheduler

def get_callbacks(settings, log = False):
    #CSV Logger
    if log:
        log_dir = f'../../reports/{settings.dataset}/{settings.save_name}/logs/expert'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        csv_logger = CSVLogger(f'{log_dir}/{settings.log_name}_history.csv')

    #Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    callbacks = [early_stopping]

    if settings.learning_rate_scheduler:
        if args.learning_rate_scheduler == 'Linear':
            lr_scheduler = LearningRateScheduler(linear_decay)
            callbacks.append(lr_scheduler)

    return callbacks

def save_to_csv(save_name, best_val_loss):
    # Check if the CSV file exists
    file_exists = os.path.isfile('model_trial_history.csv')
    
    # Append the results to the CSV
    with open('model_trial_history.csv', 'a') as f:
        if not file_exists:
            f.write("ModelName,BestValLoss\n")  # Write header if file doesn't exist
        f.write(f"{save_name},{best_val_loss}\n")

class DeadModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, optimizer, learning_rate, death_frequency_check=20, revival_epochs=5):
        super(DeadModelCallback, self).__init__()
        self.death_frequency_check = death_frequency_check
        self.revival_epochs = revival_epochs
        self.n_resuscitation = 0
        self.resuscitation_limit = 100  # You can adjust this as needed
        self.dead_epochs_count = 0
        self.validation_data = validation_data
        self.alive = True
        self.optimizer = optimizer
        self.initial_learning_rate = learning_rate

    def on_epoch_end(self, epoch, logs=None):
        # Check for dead model only every death_frequency_check epochs
        if (epoch + 1) % self.death_frequency_check == 0 or not self.alive:
            print("Checking if the model is dead!")
            y_pred = self.model.predict(self.validation_data[0])

            # Check if the model is "dead"
            if np.var(y_pred) < 1e-10:
                self.alive = False
                self.dead_epochs_count += 1
                print(f"Warning! Potential dead model detected!")

                # If the model remains dead for revival_epochs, then reinitialize
                if self.dead_epochs_count >= self.revival_epochs:
                    self.n_resuscitation += 1

                    if self.n_resuscitation > self.resuscitation_limit:
                        raise ValueError('Unsuccessful resuscitation, check the architecture.')

                    print(f"Warning! Dead model at epoch! Reinitiating ({self.n_resuscitation}/{self.resuscitation_limit})...")

                    # Reinitialize model weights
                    for layer in self.model.layers:
                        if hasattr(layer, 'kernel_initializer'):
                            layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape))
                            if layer.use_bias:
                                layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape))
                    self.optimizer.learning_rate = self.initial_learning_rate
                    self.dead_epochs_count = 0  # Reset the counter
                    self.alive = True
            else:
                self.dead_epochs_count = 0  # Reset the counter if model is not dead
                self.alive = True


def linear_decay(epoch, lr):
    # Assuming you want to decay over 100 epochs, adjust if different
    return lr * (1 - epoch / 200.0)