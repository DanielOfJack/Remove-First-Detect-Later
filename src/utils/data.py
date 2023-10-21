import numpy as np
import pickle

def split_into_patches(arr, N=32):  # Default N to 32 for 32x32 patches
    # Number of images
    num_images = arr.shape[0]
    
    # Number of patches along one dimension of the image
    num_patches = 512 // N
    
    # Create an empty array to hold all patches
    patches = np.empty((num_images, num_patches, num_patches, N, N, 1), dtype=arr.dtype)
    
    for i in range(num_patches):
        for j in range(num_patches):
            patches[:, i, j] = arr[:, i*N:(i+1)*N, j*N:(j+1)*N, :]
    
    # Reshape to have all patches in a single dimension
    patches = patches.reshape(num_images * num_patches * num_patches, N, N, 1)
    
    return patches

def reconstruct_from_patches(patches, N=32):  # Default N to 32 for 32x32 patches
    # Number of images
    num_images = patches.shape[0] // (512 // N)**2
    
    # Create an empty array to hold the reconstructed images
    reconstructed = np.empty((num_images, 512, 512, 1), dtype=patches.dtype)
    
    num_patches = 512 // N
    
    for img_idx in range(num_images):
        for i in range(num_patches):
            for j in range(num_patches):
                reconstructed[img_idx, i*N:(i+1)*N, j*N:(j+1)*N, :] = patches[img_idx*num_patches*num_patches + i*num_patches + j]
                
    return reconstructed

def normalize_patches(patches):
    # Compute the minimum and maximum values for each patch
    patch_mins = patches.min(axis=(1, 2, 3), keepdims=True)
    patch_maxs = patches.max(axis=(1, 2, 3), keepdims=True)
    
    # Avoid division by zero by adding a small value
    epsilon = 1e-7
    patch_maxs = np.where(patch_maxs == patch_mins, patch_mins + epsilon, patch_maxs)
    
    # Apply min-max scaling
    normalized_patches = (patches - patch_mins) / (patch_maxs - patch_mins)
    
    return normalized_patches

def load_pickle(data_path, limit=1500, only="Both", dataset="LOFAR"):
    
    print(f"LOADING LOFAR DATASET FROM {data_path}")
    
    (train_data, train_masks, test_data, test_masks) = np.load(f'{data_path}', allow_pickle=True)
    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max
    
    print("SUCCESS")
    
    if limit is not None:
        np.random.seed(42)
        train_indx = np.random.permutation(len(train_data))[:limit]
        train_data  = train_data[train_indx]
        train_masks = train_masks[train_indx]
        np.random.seed(None)
        
    print("PREPROCESSING...")
    
    train_data = preprocess(train_data, dataset)
    test_data = preprocess(test_data, dataset)
    
    print("PREPROCESSING COMPLETE")

    if only=="Train":
        return train_data.astype('float32'), train_masks.astype(np.int16)
    elif only=="Test":
        return test_data.astype('float32'), test_masks.astype(np.int16)
    return train_data.astype('float32'), train_masks, test_data.astype('float32'), test_masks

def preprocess_spectrogram(amplitude, dataset):
    clip_value = 20
    if (dataset == "HERA"):
        clip_value = 4
    
    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    amplitude = np.clip(amplitude, mean_amp - std_amp, mean_amp + clip_value * std_amp)

    # Take the natural logarithm
    amplitude = np.log(amplitude + 1e-16)  # added a small value to avoid log(0)

    # MinMax scale the amplitude
    min_amp = np.min(amplitude)
    max_amp = np.max(amplitude)
    if max_amp == min_amp:
        amplitude = 0
    else:
        amplitude = (amplitude - min_amp) / (max_amp - min_amp)
    return amplitude

def preprocess(X, dataset):
    N = X.shape[0]
    preprocessed_data = np.zeros_like(X)
    
    for i in range(N):
        preprocessed_data[i] = preprocess_spectrogram(X[i], dataset)
        
    return preprocessed_data