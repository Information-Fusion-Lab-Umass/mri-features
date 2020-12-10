import nibabel as nib
import numpy as np
import scipy as sp

def center_crop(img, length, width, height):
    x = img.shape[0]//2 - length//2
    y = img.shape[1]//2 - width//2
    z = img.shape[2]//2 - height//2
    return img[x:x+length, y:y+width, z:z+height]

def random_crop(img, length, width, height):
    x = np.random.randint(0, img.shape[0] - length)
    y = np.random.randint(0, img.shape[1] - width)
    z = np.random.randint(0, img.shape[2] - height)
    return img[x:x+length, y:y+width, z:z+height]

def augment_image(image):
    sigma = np.random.uniform(0.0,1.0,1)[0]
    return sp.ndimage.filters.gaussian_filter(image, sigma, truncate=8)

def load_mri(path):
    #logging.debug(f'loading image from {path}')
    img = nib.load(path).get_fdata(dtype=np.float32)
    
    # Replace nans in image with 0
    img[np.isnan(img)] = 0

    # Standardize pixel values
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    # Crop the image to match the input dim of the model
    # TODO: Make use of augment_image / random_crop for training
    img = center_crop(img, 96, 96, 96)

    return np.expand_dims(img, axis=0)
