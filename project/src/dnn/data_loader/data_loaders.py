import numpy as np
import torch
from torch.utils.data import TensorDataset

from base import BaseDataLoader


class DNNDataLoader(BaseDataLoader):
    """
    Data loading using BaseDataLoader.
    """
    def __init__(self,
                 X_train_path,
                 Y_train_path,
                 batch_size,
                 shuffle=True,
                 validation_split=0.0):
        x_train = torch.from_numpy(np.load(X_train_path)['arr_0']).float()
        y_train = torch.from_numpy(np.load(Y_train_path)['arr_0']).float()
        self.dataset = TensorDataset(x_train, y_train)
        super(DNNDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split)
