import torch
from torch.utils import data
import os
import numpy as np


class TumorDataset(data.Dataset):
    """
    Tumor Simulation Dataset
    """

    def __init__(self, ids, save_path):
        self.ids = ids
        self.save_path = save_path

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def normalize(y):
        # normalize with the mean and standard deviation of the whole dataset
        # return (y - 0.0106) / 0.0911
        return (y - 0.0018) / 6.5622

    def __getitem__(self, index):
        id = self.ids[index]

        # load the file
        raw_data = np.load(os.path.join(self.save_path, id + '.npz'))

        # the input is the 6 input variables
        X = torch.Tensor(raw_data['y'])

        # the ground is the 3d volume
        y = torch.Tensor(raw_data['x'][:, :, :, 0])

        # normalize the 3d volume
        y = self.normalize(y)

        # unsqueeze to add one channel
        y = torch.unsqueeze(y, 0)

        return X, y
