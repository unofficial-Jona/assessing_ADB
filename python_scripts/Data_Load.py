import warnings
warnings.warn('add location of dataset and annotations_file')


# import torch
# import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np
import pandas as pd


class DrivingData(Dataset):
    def __init__(self, annotations_file, img_dir):
        # implement default for img_dir and annotations_file
        self.img_labels = pd.read_csv(annotations_file) 
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        return image, label
    

def split_data(train_split=0.8, seed = 42):
    """Spliting data into train and test sets 

    Args:
        train_split (float [0,1], optional): relative size of the train set. Set to 1 to train on entire set. Defaults to 0.8.
        seed (int, optional): seed value for reproducibility. Defaults to 42.

    Returns:
        pytorch Dataset: train_set, test_set
    """
    dataset = DrivingData()
    train_size = np.floor(dataset.__len__() * train_split)
    test_size = dataset.__len__() - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    return train_set, test_set

