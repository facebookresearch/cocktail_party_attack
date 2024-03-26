#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module for dataset related functions"""
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from io import open
from torchvision.datasets import ImageFolder
from os.path import expanduser
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
import pickle
from torchtext.vocab import GloVe

torch.multiprocessing.set_sharing_strategy("file_system")
home = expanduser("~")
cwd = os.getcwd()
ds_root = f"{cwd}/datasets"


def read_pickle(pkl_file):
    with open(pkl_file, "rb") as handle:
        data = pickle.load(handle)
    return data


def write_pickle(data, pkl_file):
    with open(pkl_file, "wb") as handle:
        data = pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TinyImageNet200(ImageFolder):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        root=f"{ds_root}/tiny-imagenet-200",
        **kwargs,
    ):
        subfolder = "train" if train else "val"
        root_sub = os.path.join(root, subfolder)

        if not os.path.exists(root):
            raise ValueError(
                "Dataset not found at {}. Please download it from {}.".format(
                    root, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )
            )

        super().__init__(
            root=root_sub,
            transform=transform,
            target_transform=target_transform,
        )


class ImageNet(ImageFolder):
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
        root=f"{ds_root}/imagenet",
        **kwargs,
    ):
        subfolder = "train" if train else "val"
        root_sub = os.path.join(root, subfolder)

        if not os.path.exists(root):
            raise ValueError("Dataset not found at {}.")

        super().__init__(
            root=root_sub,
            transform=transform,
            target_transform=target_transform,
        )


nclasses_dict = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
    "imagenet": 1000,
}

xshape_dict = {
    "cifar10": [3, 32, 32],
    "cifar100": [3, 32, 32],
    "tiny_imagenet": [3, 64, 64],
    "imagenet": [3, 224, 224],
}


ds_dict = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "tiny_imagenet": TinyImageNet200,
    "imagenet": ImageNet,
}

ds_dir_dict = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "tiny_imagenet": "tiny-imagenet-200",
    "imagenet": "imagenet",
}

ds_type_dict = {
    "cifar10": "image",
    "cifar100": "image",
    "tiny_imagenet": "image",
    "imagenet": "image",
}

task_type_dict = {
    "cifar10": "classification",
    "cifar100": "classification",
    "imagenet": "classification",
    "tiny_imagenet": "classification",
}

transforms_noaugment = {
    "cifar10": [],
    "cifar100": [],
    "tiny_imagenet": [],
    "imagenet": [
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
    ],
}

transforms_augment = {
    "cifar10": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "cifar100": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ],
    "tiny_imagenet": [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ],
    "imagenet": [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ],
}

ds_mean = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4867, 0.4408],
    "tiny_imagenet": [0.485, 0.456, 0.406],
    "imagenet": [0.485, 0.456, 0.406],
}

ds_std = {
    "cifar10": [0.2023, 0.1994, 0.2010],
    "cifar100": [0.2675, 0.2565, 0.2761],
    "tiny_imagenet": [0.229, 0.224, 0.225],
    "imagenet": [0.229, 0.224, 0.225],
}


def get_dataloaders(
    ds: str = "cifar10",
    batch_size: int = 256,
    augment: bool = True,
    shuffle_train=True,
    shuffle_test=False,
    num_workers=8,
):
    """returns train and test loaders"""

    transform_train_list, transform_test_list = [transforms.ToTensor()], [
        transforms.ToTensor()
    ]

    if augment:
        transform_train_list += transforms_augment[ds]
    else:
        transform_train_list += transforms_noaugment[ds]

    transform_test_list += transforms_noaugment[ds]

    transform_train_list += [transforms.Normalize(mean=ds_mean[ds], std=ds_std[ds])]
    transform_test_list += [transforms.Normalize(mean=ds_mean[ds], std=ds_std[ds])]

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    dataset_train = ds_dict[ds](
        root=f"{ds_root}/{ds_dir_dict[ds]}",
        train=True,
        transform=transform_train,
        download=True,
    )
    dataset_test = ds_dict[ds](
        root=f"{ds_root}/{ds_dir_dict[ds]}",
        train=False,
        transform=transform_test,
    )

    dl_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dl_train, dl_test
