# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import itertools
import torch.utils.data
import torchvision.transforms as transforms

from search.data_providers.base_provider import *
from search.datasets.tumor_dataset import TumorDataset


class TumorDataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None,
                 n_worker=8, dims=3, output_size=128):

        self.patient_id = [i for i in range(1)]
        self.parameters = [i for i in range(0, 100)]
        self.timesteps = [i for i in range(1, 21)]

        self.ids = self.get_ids(self.patient_id, self.parameters, self.timesteps)
        self.dims = dims
        self.output_size = output_size

        self._save_path = save_path
        train_dataset = TumorDataset(self.ids, self._save_path, self.dims, self.output_size)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size

            indices = list(range(len(train_dataset)))
            train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

            valid_dataset = TumorDataset(self.ids, self._save_path, self.dims, self.output_size)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = TumorDataset(self.ids, self._save_path, self.dims, self.output_size)

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'tumor_simul'

    @property
    def data_shape(self):
        return 6,

    @property
    def n_classes(self):
        return None

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/qasima/proxylessnas/dataset/tumor_simul'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download tumor_simul')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize((0,), (1,))

    @staticmethod
    def get_ids(patient_id, parameters, timesteps):
        return ["{}_{}_{}".format(x[0], x[1], x[2]) for x in itertools.product(patient_id, parameters, timesteps)]
