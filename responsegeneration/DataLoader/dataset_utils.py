import json
import os
import re
import string
import random
import warnings
from glob import glob
from itertools import chain
from operator import itemgetter
import torch
from torch.utils.data import Dataset, Sampler, DistributedSampler

from utils import *

#####################################
# Dataset/Dataloader Util Functions #
#####################################


def tokenize_dataset(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize_dataset(o, tokenizer)) for n, o in obj.items())
    return list(tokenize_dataset(o, tokenizer) for o in obj)


def get_tokenized_dataset(tokenizer, dataset_path):
    """
    Get dataset and tokenize if necessary.
    Note that MARCO is organized differently so it requires a different loading and tokenization method.
    """
    if dataset_path.endswith("tokenized.json"):
        print("Loading tokenized dataset from " + dataset_path)
        tokenize = False
    else:
        if os.path.isfile(dataset_path[:-5] + "_tokenized.json"):
            print("Detected existing tokenized file.")
            dataset_path = dataset_path[:-5] + "_tokenized.json"
            print("Loading tokenized dataset from " + dataset_path)
            tokenize = False
        else:
            print("Loading dataset from " + dataset_path)
            tokenize = True

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if tokenize:
        print("Tokenizing the dataset")
        dataset = tokenize_dataset(dataset, tokenizer)

        new_name = dataset_path[:-5] + "_tokenized.json"
        print("Saving dataset to " + new_name)
        with open(new_name, "w") as outfile:
            json.dump(dataset, outfile, indent=2)

    return dataset


def tokenize_marco(obj, tokenizer):
    """
    Function specifically to tokenize our version of the MARCO dataset.
    """
    for i in range(len(obj["context"])):
        obj["context"][i]["passage_text"] = tokenizer.encode(
            obj["context"][i]["passage_text"]
        )
    for i in range(len(obj["utterances"])):
        obj["utterances"][i]["history"] = [
            tokenizer.encode(x.lstrip(" ()_")) for x in obj["utterances"][i]["history"]
        ]
        for j in range(len(obj["utterances"][i]["candidates"])):
            obj["utterances"][i]["candidates"][j] = tokenizer.encode(
                obj["utterances"][i]["candidates"][j]
            )


class StatefulSampler(Sampler):
    """
    Vector Institute's Dataloader sampler for saving state between preemptions.
    This one works fine for multiple choice classification loss, since there isn't any restriction on
    how inputs are shuffled.
    """

    def __init__(self, data_source, shuffle=False):
        self.data = data_source
        self.shuffle = shuffle

        # initial dataloader index
        self.init_index()

    def init_index(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        else:
            self.indices = torch.arange(len(self.data))
        self.data_counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.data_counter == len(self.data):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = (
                    dataloader_iter._send_idx - dataloader_iter._rcvd_idx
                ) * batch_size
        return {
            "indices": self.indices,
            "data_counter": self.data_counter - prefetched_num,
        }

    def load_state_dict(self, state_dict):
        self.indices = state_dict["indices"]
        self.data_counter = state_dict["data_counter"]


class BCBatchSampler(Sampler):
    """
    When using binary classification loss, we want each batch to contain a similar amount of correct
    and incorrect instances. This is a batch sampler to do that, and should get passed in as the batch_sampler
    to the dataloader.

    IMPORTANT: uses random functions, random.seed should get set before use.
    This batch sampler probably isn't compatible with resuming training.
    """

    def __init__(self, dataset, batch_size, shuffle=True, indices=None, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # get the indicies and length
        self.correct_indices = dataset.correct_replies
        self.incorrect_indices = dataset.incorrect_replies

        if indices:
            # in cases of distributed training, indices for each rank will be passed in
            # we separate out the groups again.
            self.correct_indices = [
                value for value in indices if value in self.correct_indices
            ]
            self.incorrect_indices = [
                value for value in indices if value in self.incorrect_indices
            ]

        self.num_items = len(self.correct_indices) + len(self.incorrect_indices)

        # number of batches (i just brute force this)
        self.num_batches = 0
        batch_num = 0
        for idx in range(self.num_items):
            if idx < len(self.correct_indices):
                batch_num = batch_num + 1
                if batch_num == self.batch_size:
                    batch_num = 0
                    self.num_batches = self.num_batches + 1
            else:
                break

            # add incorrect item
            if idx < len(self.incorrect_indices):
                batch_num = batch_num + 1
                if batch_num == self.batch_size:
                    batch_num = 0
                    self.num_batches = self.num_batches + 1
            else:
                break
        if not self.drop_last and batch_num > 0:
            self.num_batches = self.num_batches + 1

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.correct_indices)
            random.shuffle(self.incorrect_indices)

        batch = []
        for idx in range(self.num_items):
            # add correct item
            if idx < len(self.correct_indices):
                batch.append(self.correct_indices[idx])
                if len(batch) == self.batch_size:
                    random.shuffle(batch)
                    yield batch
                    batch = []
            else:
                break

            # add incorrect item
            if idx < len(self.incorrect_indices):
                batch.append(self.incorrect_indices[idx])
                if len(batch) == self.batch_size:
                    random.shuffle(batch)
                    yield batch
                    batch = []
            else:
                break

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return self.num_batches


class DistributedBCBatchSampler(DistributedSampler):
    """
    Wrapper for the BCBatchSampler in the case of distributed training.
    This should get passed in as the batch sampler to the dataloader.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=10,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BCBatchSampler(
            self.dataset, batch_size=self.batch_size, indices=indices, shuffle=True
        )
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
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


class DistributedSamplerWrapper(DistributedSampler):
    """
    Borrowed from https://github.com/pytorch/pytorch/issues/23430
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

    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True):
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
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

    def state_dict(self, dataloader_iter=None):
        return self.sampler.state_dict(dataloader_iter)

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict)


def padding_collate_fn(batch):
    """
    Collate function for dataloader on dataset with MC second head loss.
    Exists to pad inputs to same length within the batch.
    """
    instance = {}
    pad_token = batch[0]["pad_token"]
    for key in batch[0].keys():
        if key in PADDED_INPUTS:
            max_len = max([len(y) for x in batch for y in x[key]])
            # max_len = 350  # for testing what the max length should be to fit on gpu
            new_shape = (len(batch), len(batch[0]["input_ids"]), max_len)
            padded = [
                x + [pad_token if key != "lm_labels" else -100] * (max_len - len(x))
                for y in batch
                for x in y[key]
            ]
            instance[key] = torch.Tensor(padded).reshape(new_shape)
        else:
            # print(key, batch[0][key])
            instance[key] = torch.Tensor([x[key] for x in batch])
    return instance


def padding_bc_collate_fn(batch):
    instance = {}
    pad_token = batch[0]["pad_token"]
    for key in batch[0].keys():
        if key in PADDED_INPUTS:
            max_len = max([len(x[key]) for x in batch])
            padded = [
                x[key]
                + [pad_token if key != "lm_labels" else -100] * (max_len - len(x[key]))
                for x in batch
            ]
            instance[key] = torch.Tensor(padded)
        else:
            instance[key] = torch.Tensor([x[key] for x in batch])
    return instance
