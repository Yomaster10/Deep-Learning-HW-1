import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler
import random


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        N = self.__len__()
        idx = []
        for i in range(N):
            idx.append(i)
            if N-i-1 > idx[-1]:
                idx.append(N-i-1)
            else:
                break
            if i+1 == idx[-1]:
                break
        for j in idx:
            yield j
        # ========================

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    idx = list(range(len(dataset)))
    val_len = int(np.floor(validation_ratio * len(dataset)))

    val_idx = np.random.choice(idx, val_len, replace=False)
    train_idx = [i for i in idx if i not in val_idx]
    
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    # ========================

    return dl_train, dl_valid
