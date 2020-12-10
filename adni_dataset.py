import numpy as np
from torch.utils.data import Dataset

import image_utils

class AdniDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.categories = ['CN', 'AD', 'MCI']
        self.labels = np.array([self.encode_label(label) for label in df['label2']])
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        img = image_utils.load_mri(x['caps_path'])
        label = self.labels[idx]
        return idx, img, label

    def encode_label(self, label):
        assert label in self.categories
        # CrossEntropyLoss expects categorical indices, *not* one-hot encoded labels. It also expects int64 input
        return np.int64(self.categories.index(label))
