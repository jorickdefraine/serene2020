import os

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms


def get_data_loaders(data_dir, batch_size, valid_size, shuffle, num_workers, pin_memory, random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.ImageNet DataSet.
    :param data_dir: Location of the DataSet. Downloading not supported.
    :param batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = os.path.join(data_dir, "ImageNet")

    train_dataset = datasets.ImageNet(
        root=data_dir, split="train", transform=transform_train,
    )

    if valid_size > 0:
        valid_dataset = datasets.ImageNet(
            root=data_dir, split="train", transform=transform_test,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    test_dataset = datasets.ImageNet(
        root=data_dir, split="val", transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader, test_loader) if valid_size > 0 else (train_loader, test_loader)
