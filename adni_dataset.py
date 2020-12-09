import logging

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class AdniDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.categories = ['CN', 'AD', 'MCI']
        self.labels = np.array([self.encode_label(label) for label in df['label2']])
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        img = self.load_image(x['caps_path'])
        label = self.labels[idx]
        return idx, img, label

    def load_image(self, path):
        logging.debug(f'loading image from {path}')
        img = nib.load(path).get_fdata()
        
        # Replace nans in image with 0
        img[img != img] = 0

        # Crop the image from 121 x 145 x 121 to 96 x 96 x 96
        old_shape = np.array(img.shape)
        new_shape = np.array([96, 96, 96])
        start = np.floor((old_shape - new_shape) / 2).astype(int)
        end = start + new_shape
        img = img[start[0]: end[0], start[1]: end[1], start[2]: end[2]]

        # The CNN expects 4D, float32 input
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        return img
        
    def encode_label(self, label):
        assert label in self.categories
        # CrossEntropyLoss expects categorical indices, *not* one-hot encoded labels. It also expects int64 input
        return np.int64(self.categories.index(label))
