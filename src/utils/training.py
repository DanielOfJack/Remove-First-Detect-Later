import numpy as np

def corrupt_images(images):
    corrupted_images = images.copy()
    img_height, img_width = images.shape[1:3]

    for i in range(images.shape[0]):

        # Add 0 to 3 dashed lines of corruption
        for _ in range(np.random.randint(1, 20)):
            brightness_factor = np.random.uniform(0.15, 0.55)
            orientation = np.random.choice(['horizontal', 'vertical'])
            line_width = np.random.randint(1, 2)
            dash_length = np.random.randint(1, img_width + 1)
            gap_length = np.random.randint(1, img_width + 1)
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
            brightness_factor = np.random.uniform(0.15, 0.55)
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
    
    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_masks = np.concatenate(augmented_masks, axis=0)
    
    return augmented_data, augmented_masks