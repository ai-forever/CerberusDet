import random
from operator import itemgetter
from typing import Iterator, Optional

import numpy as np
import torch


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, class_choice="least_sampled"):
        """
        class_choice: a string indicating how class will be selected for every
        sample:
            "least_sampled": class with the least number of sampled labels so far
            "random": class is chosen uniformly at random
            "cycle": the sampler cycles through the classes sequentially
        """

        self.labels = [None] * len(dataset)  # example labels
        self.class_indices = {}  # lists of example indices per class

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            index = dataset.indices[idx]
            if not isinstance(dataset, torch.utils.data.Subset):
                assert index == idx
                if dataset.labels[index].shape[1] == 6:
                    labels = dataset.labels[index][:, 0]
                else:
                    assert dataset.labels[index].shape[1] == 7
                    # assumed that there are less than 50 classes for each task
                    labels = dataset.labels[index][:, 1] + (dataset.labels[index][:, 0] * 50)
            else:
                assert dataset.dataset.labels[index].shape[1] == 6
                labels = dataset.dataset.labels[index][:, 0]
            # TODO: adapt for joined dataset
            labels = list(map(int, labels.tolist()))
            for label in labels:
                if label not in self.class_indices:
                    self.class_indices[label] = []
                self.class_indices[label].append(idx)
            self.labels[idx] = labels

        all_classes = list(self.class_indices.keys())
        self.all_classes = list(map(int, all_classes))
        self.counts = {cl: 0 for cl in self.all_classes}

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        self.counts = {cl: 0 for cl in self.all_classes}
        return self

    def __next__(self):
        if self.count >= len(self.labels):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = random_choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_ in self.labels[chosen_index]:
                self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            ind = random.randint(0, len(self.all_classes) - 1)
            class_ = self.all_classes[ind]
        elif self.class_choice == "cycle":
            class_ = self.all_classes[self.current_class]
            self.current_class = self.current_class + 1
            if self.current_class >= len(self.all_classes):
                self.current_class = 0
        elif self.class_choice == "least_sampled":
            first_key = list(self.counts.keys())[0]
            min_count = self.counts[first_key]
            min_classes = [first_key]
            for class_ in self.all_classes:
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = random_choice(min_classes)
        return class_

    def __len__(self):
        return len(self.labels)


def random_choice(list_data):
    # chosen_index = np.random.choice(class_indices)
    i = np.random.randint(0, len(list_data))
    return list_data[i]


class DatasetFromSampler(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class RepeatSampler(object):
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
