import numpy as np
import argparse
import tensorflow as tf
import os
import pickle
import pandas as pd
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from architectures.convolutional import Convolutional
from sklearn.model_selection import train_test_split
from utils.training import augment
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from keras import backend as K
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter
from architectures.RFDL import weighted_mse
from utils.data import load_pickle, split_into_patches, normalize_patches

def get_RFDL(X_train, y_train, dataset):
        X_train = (split_into_patches(X_train, 64))
        y_train = split_into_patches(y_train, 64)
        gc.collect()
                
        remove = load_model(f'models/{dataset}/CDAE.h5', custom_objects={'weighted_mse': weighted_mse})
        X_removed = smooth_images(remove.predict(X_train))
        X_train = X_train - X_removed
        return X_train, y_train

def get_expert_train_test(X_test, y_test, args):
    PATCH_SZ = 512
    TRAIN_SZ = 20
    VAL_SZ = 19
    
    if args.dataset == "HERA":
        VAL_SZ = 100
    
    X_train, y_train = shuffle(X_test, y_test, random_state=args.seed)
    
    if args.model_name == 'RFDL':
        X_train, y_train = get_RFDL(X_train, y_train, args.dataset)
        PATCH_SZ = 64
        VAL_SZ = VAL_SZ*64
        TRAIN_SZ = TRAIN_SZ*64
    
    X_train_trial, y_train_trial = augment(X_train[:TRAIN_SZ], y_train[:TRAIN_SZ])
    X_val_trial = X_train[TRAIN_SZ:(TRAIN_SZ+VAL_SZ)]
    X_test_trial = X_train[(TRAIN_SZ+VAL_SZ):]
    y_val_trial = y_train[TRAIN_SZ:(TRAIN_SZ+VAL_SZ)]
    y_test_trial = y_train[(TRAIN_SZ+VAL_SZ):]
    
    return X_train_trial, y_train_trial, X_val_trial, y_val_trial, X_test_trial, y_test_trial

def smooth_images(images):
    smoothed_images = np.empty_like(images)
    
    for i in range(images.shape[0]):
        smoothed_images[i] = gaussian_filter(images[i], sigma=2.5)
    
    return smoothed_images

def train_convolutional(save_name, model_name, data_name, loss, X_train, y_train, X_val, y_val, epochs, labels, save=True):
    
    patch_sz = 512
    if model_name == 'RFDL':
        patch_sz = 64
    
    conv = Convolutional(
    save_name = save_name,
    model_name = model_name,
    patch_sz = patch_sz,
    epochs=epochs,
    loss = loss,
    )
    
    model = conv.compile_model()
    model, score = conv.train_model(model, X_train, y_train, X_val, y_val)
    if (save):
        model.save(f'models/{data_name}/{model_name}_{labels}.h5')
        # conv.plot_training_history()
    return model

def main(args):
    if args.save_name is None:
        args.save_name = args.model_name
    
    data_map = {
        "LOFAR": "../data/LOFAR_Full_RFI_dataset.pkl",
        "HERA": "../data/HERA-SIM_Full_RFI_dataset.pkl"
    }
    data_path = data_map[args.dataset]

    #Load Training Data
    X_train, y_train, X_test, y_test = load_pickle(data_path, limit=1500, dataset = args.dataset)
    
    if (args.labels == 'Expert20'):
        X_train_trial, y_train_trial, X_val_trial, y_val_trial, X_test_trial, y_test_trial = get_expert_train_test(X_test, y_test, args)
        X_train, y_train = None, None
        gc.collect()
    else:
        X_train, y_train = shuffle(X_train, y_train, random_state=args.seed)
        gc.collect()
    
        if args.model_name == 'RFDL':
            X_train, y_train = get_RFDL(X_train, y_train, args.dataset)
            X_test, y_test = get_RFDL(X_test, y_test, args.dataset)

        X_train_trial, X_val_trial, y_train_trial, y_val_trial = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
        X_test_trial = X_test
        y_test_trial = y_test
        
    print("TRAINING")
    model = train_convolutional(args.save_name, args.model_name, args.dataset, args.loss, X_train_trial, y_train_trial, X_val_trial, y_val_trial, args.epochs, args.labels, save=True)
    
    if args.report:
        print("TESTING")
        y_pred_probs = model.predict(X_test_trial).reshape(-1,1)
        y_test_trial = y_test_trial.reshape(-1,1)

        # Calculate metrics
        auroc = roc_auc_score(y_test_trial, y_pred_probs)
        auprc = average_precision_score(y_test_trial, y_pred_probs)

        # Calculate F1 score at a threshold of 0.5
        y_pred_05 = (y_pred_probs > 0.5).astype(int)
        f1_05 = f1_score(y_test_trial, y_pred_05)

        # Calculate best possible F1 score
        precision, recall, thresholds = precision_recall_curve(y_test_trial, y_pred_probs)

        denominator = precision + recall

        # Initialize f1_scores with zeros
        f1_scores = np.zeros_like(denominator)

        # Update only where denominator is not zero
        mask = denominator != 0
        f1_scores[mask] = (2 * precision[mask] * recall[mask]) / denominator[mask]

        best_f1 = f1_scores.max()

        # Save metrics to CSV
        results = pd.DataFrame({
            'MODEL': [args.save_name],
            'AUROC': [auroc],
            'AUPRC': [auprc],
            'Best F1 Score': [best_f1],
            'F1 Score': [f1_05]
        })

        write_header = True

        dir_path = f'report/{args.dataset}/'

        # Check if the directory exists, and if not, create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Check if the CSV file exists
        file_path = f'report/{args.dataset}/{args.labels}_Labels.csv'
        write_header = not os.path.exists(file_path)

        # Append results to CSV and write header only if the file doesn't exist
        results.to_csv(file_path, mode='a', header=write_header, index=False)
        print("TRIAL COMPLETE: RESULTS SAVED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model with specified hyperparameters.')
    parser.add_argument('model_name', type=str, default="RNET7", help='Name of Save File')
    parser.add_argument('--save_name', type=str, default=None, help='Name of Save File')
    parser.add_argument('--loss', choices=['mse', 'binary_crossentropy'], default="mse", help='Loss to be used')
    parser.add_argument('--dataset', choices=['LOFAR', 'HERA'], default='LOFAR', help='Dataset to be used.')
    parser.add_argument('--labels', choices=['AOFlagger', 'Expert20'], default='Expert20', help='Type of labels to use for experiment')
    parser.add_argument('--seed', type=int, default=None, help='Seed for data split')
    parser.add_argument('--epochs', type=int, default=None, help='Seed for data split')
    parser.add_argument('--report', type=int, default=1, help='Report results (1: Yes, 0: No')
    args = parser.parse_args()

    main(args)  # Pass the args object to the main function