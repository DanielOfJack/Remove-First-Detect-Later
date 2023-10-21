from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, Callback, LearningRateScheduler
from tensorflow.keras.initializers import TruncatedNormal, glorot_uniform
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import argparse

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as pltsplit_into_patchessplit_into_patches
from utils.data import load_pickle, split_into_patches, normalize_patches
from architectures.RFDL import CDAE

def corrupt_images(images):
    corrupted_images = images.copy()
    img_height, img_width = images.shape[1:3]

    for i in range(images.shape[0]):
        
        for _ in range(np.random.randint(0, 4)):
            brightness_factor = np.random.uniform(0.05, 0.20)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 4)

            if orientation == 'horizontal':
                x = np.random.randint(0, img_width - line_width + 1)
                corrupted_images[i, x:x+line_width, :, :] += brightness_factor
            else:
                y = np.random.randint(0, img_height - line_width + 1)
                corrupted_images[i, :, y:y+line_width, :] += brightness_factor

        # Add 0 to 3 dashed lines of corruption
        for _ in range(np.random.randint(1, 10)):
            brightness_factor = np.random.uniform(0.10, 0.50)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 2)
            dash_length = np.random.randint(1, img_width/2 + 1)
            gap_length = np.random.randint(1, img_width/4 + 1)
            if orientation == 'horizontal':
                x = np.random.randint(0, img_width - line_width + 1)
                y = 0
                while y < img_height:
                    corrupted_images[i, x:x+line_width, y:y+dash_length, :] += brightness_factor
                    y += dash_length + gap_length
            else:
                x = 0
                y = np.random.randint(0, img_height - line_width + 1)
                while x < img_width:
                    corrupted_images[i, x:x+dash_length, y:y+line_width, :] += brightness_factor
                    x += dash_length + gap_length

        # Add 0 to 3 small blocks of corruption
        for _ in range(np.random.randint(1, 3)):
            brightness_factor = np.random.uniform(0.10, 0.50)
            block_size = np.random.randint(2, 5)
            x = np.random.randint(0, img_width - block_size + 1)
            y = np.random.randint(0, img_height - block_size + 1)
            corrupted_images[i, x:x+block_size, y:y+block_size, :] += brightness_factor

    
    corruption_mask = corrupted_images - images

    # Identify the pixels to corrupt further with Gaussian noise
    to_corrupt = corruption_mask > 0

    # Generate Gaussian noise
    noise = np.random.normal(0, 0.1, corrupted_images.shape)

    # Apply the Gaussian noise only to the pixels identified by the corruption mask
    corrupted_images[to_corrupt] += noise[to_corrupt]

    # Ensure pixel values remain in the valid range [0, 1]
    corrupted_images = np.clip(corrupted_images, 0, 1)

    return corrupted_images

def gaussian_noise(data):
    mean = 0
    var = 10
    sigma = var**0.5
    noise = np.random.normal(mean, sigma, data.shape)
    noisy_data = np.clip(data + noise, 0, 1)  # Clip values to be in valid range
    return noisy_data

def flip(data, flip_type):
    if flip_type == 0:
        return np.flip(data, axis=1)  # Vertical flip
    elif flip_type == 1:
        return np.flip(data, axis=2)  # Horizontal flip
    elif flip_type == 2:
        return np.flip(np.flip(data, axis=1), axis=2)  # Both flips

def rotate(data, k):
    return np.rot90(data, k=k, axes=(1, 2))

def augment(data, mask):
    augmented_data = []
    augmented_masks = []

    for flip_type in range(3):  # 0: vertical, 1: horizontal, 2: both
        data_flipped = flip(data, flip_type)
        mask_flipped = flip(mask, flip_type)
        
        for k in range(4):  # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees
            data_rotated = rotate(data_flipped, k)
            mask_rotated = rotate(mask_flipped, k)
            
            augmented_data.append(data_rotated)
            augmented_masks.append(mask_rotated)
    
    # Adding Gaussian noise to the augmented data
    # noisy_data = gaussian_noise(np.concatenate(augmented_data, axis=0))
    
    # Concatenating the noisy data with the original augmented data
    augmented_data = np.concatenate(augmented_data, axis=0)
    
    # Repeating the masks for the noisy data
    augmented_masks = np.concatenate(augmented_masks, axis=0)
    
    return augmented_data, augmented_masks

def main(dataset):
    batch_sz = 64
    
    data_map = {
        "LOFAR": "../data/LOFAR_Full_RFI_dataset.pkl",
        "HERA": "../data/HERA-SIM_Full_RFI_dataset.pkl"
    }
    data_path = data_map[args.dataset]

    #Load Training Data
    X_train, y_train, X_test, y_test = load_pickle(data_path, limit=1500)
    X_train = (split_into_patches(X_train, batch_sz))
    y_train = split_into_patches(y_train, batch_sz)

    print("ADDING SYNTHETIC CORRUPTION...")
    normal_patches = np.where(np.all(y_train == 0, axis=(1,2,3)))[0]
    X_train_normal = X_train[normal_patches]
    X_train_normal = np.concatenate((X_train_normal, flip(X_train_normal,1)), axis=0)

    X_train_corrupted = corrupt_images(X_train_normal)

    X_noise = (X_train_corrupted - X_train_normal) > 0
    y_train_combined = np.concatenate([X_train_normal, X_noise], axis=-1)


    # Shuffle the indices to get a random sample of images
    shuffled_indices = np.random.permutation(len(X_train_normal))[:8]

    # Plot the actual, corrupted, and inpainted images side by side
    fig, axes = plt.subplots(8, 3, figsize=(24, 24))

    # Adjust the margins between the subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i, idx in enumerate(shuffled_indices):
        # Plot the actual image
        axes[i, 0].imshow(X_train_normal[idx].reshape(batch_sz, batch_sz), cmap='magma', vmin=0, vmax=1)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'Normal Patch {idx}')

        # Plot the actual image
        axes[i, 1].imshow(X_train_corrupted[idx].reshape(batch_sz, batch_sz), cmap='magma', vmin=0, vmax=1)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Corrupted Patch {idx}')

        # Plot the actual image
        axes[i, 2].imshow((X_train_corrupted[idx] - X_train_normal[idx]).reshape(batch_sz, batch_sz), cmap='magma', vmin=0, vmax=1)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f'Corruption Mask {idx}')

    plt.tight_layout()
    if not os.path.exists(f"report/{dataset}/"):
            os.makedirs(f"report/{dataset}/")
    plt.savefig(f"report/{dataset}/Synthetic_Corruption_Example.png")

    print("TRAINING CDAE...")

    cdae = CDAE((64,64,1))
    checkpoint = ModelCheckpoint(f'models/{dataset}checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    cdae.fit(X_train_corrupted, y_train_combined, epochs=120, batch_size=128, validation_split=0.2, verbose=1, callbacks=[checkpoint])
    cdae.save(f'../models/{dataset}/CDAE.h5')

    print("MODEL SAVED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for CDAE Training")
    parser.add_argument('--dataset', choices=['LOFAR', 'HERA'], default='LOFAR', help='Dataset to be used for training')

    args = parser.parse_args()
    main(args.dataset)
