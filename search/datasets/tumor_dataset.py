import torch
from torch.utils import data
import os
import numpy as np


class TumorDataset(data.Dataset):
    """
    Tumor Simulation Dataset
    """

    def __init__(self, ids, save_path, dims, output_size):
        self.ids = ids
        self.save_path = save_path
        self.dims = dims
        self.output_size = output_size

    def __len__(self):
        return len(self.ids)

    def normalize(self, y):
        # normalize with the mean and standard deviation of the whole dataset

        if self.dims == 2:
            return (y - 0.0106) / 0.0911
        elif self.dims == 3:
            return (y - 0.0018) / 6.5622

    def __getitem__(self, index):
        id = self.ids[index]

        # load the file
        raw_data = np.load(os.path.join(self.save_path, id + '.npz'))

        # the input is the 6 input variables
        # Diffusion coefficient, Proliferation rate, timestep, x, y, z
        X = torch.Tensor(raw_data['y'])

        # the ground is the 3d volume
        if self.dims == 2:
            y = torch.Tensor(raw_data['x'][:self.output_size, :self.output_size, 61, 0])
        elif self.dims == 3:
            y = torch.Tensor(raw_data['x'][:self.output_size, :self.output_size, :self.output_size, 0])

        # normalize the 3d volume
        # y = self.normalize(y)

        # unsqueeze to add one channel
        y = torch.unsqueeze(y, 0)

        return X, y * 10 ** 3
