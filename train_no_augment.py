# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image

import scipy
import zipfile
import sklearn
from sklearn import metrics
# Timing utility
from timeit import default_timer as timer

from utils import get_file_label_mapping, TestImageDataset, VGG16FineTune, train


weights = models.VGG16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

current_directory = os.getcwd()
train_dir = current_directory + "/dataset/train"
test_dir = current_directory + "/dataset/test"
valid_dir = current_directory + "/dataset/valid"

# Image transformations
image_transforms = {
    # Train uses data augmentation
    "train_augment": transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # Imagenet standards
        ]
    ),
    "train": preprocess,
    # Validation and Test does not use augmentation
    "val": preprocess,
    # "test": preprocess,
}

data = {
    "train": datasets.ImageFolder(root=train_dir, transform=preprocess),
    "train_augment": datasets.ImageFolder(
        root=train_dir, transform=image_transforms["train_augment"]
    ),
    "val": datasets.ImageFolder(root=valid_dir, transform=preprocess),
    # "test": datasets.ImageFolder(root=test_dir, transform=preprocess),
}

# NO augmentation
batch_dataloaders = {
    "train_32": DataLoader(data["train"], batch_size=32, shuffle=True),
    "train_128": DataLoader(data["train"], batch_size=128, shuffle=True),
    "val_32": DataLoader(data["val"], batch_size=32, shuffle=True),
    "val_128": DataLoader(data["val"], batch_size=128, shuffle=True),
}


checkpoints = [
    "/add-1-extra-layer-batch-32/",
    "/add-1-extra-layer-batch-128/",
]

checkpoint_dir = "checkpoint"

for checkpoint_name in checkpoints:
    if "32" in checkpoint_name:
        print("32", checkpoint_name)
        train_dataloader = batch_dataloaders["train_32"]
        val_dataloader = batch_dataloaders["val_32"]
    elif "128" in checkpoint_name:
        print("128", checkpoint_name)
        train_dataloader = batch_dataloaders["train_128"]
        val_dataloader = batch_dataloaders["val_128"]

    epochs = 30
    model = VGG16FineTune()

    train_loss, train_acc, val_loss, val_acc = train(
        model,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_dir + checkpoint_name,
    )
