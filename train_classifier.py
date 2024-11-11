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

from utils import (
    get_file_label_mapping,
    TestImageDataset,
    VGG16FineTune,
    train,
    VGG16FineTuneGeneric,
    get_model,
    data,
    checkpoint_dir,
)

batch_dataloaders = {
    "train_32": DataLoader(data["train"], batch_size=32, shuffle=True),
    "train_64": DataLoader(data["train"], batch_size=64, shuffle=True),
    "train_128": DataLoader(data["train"], batch_size=128, shuffle=True),
    "val_32": DataLoader(data["val"], batch_size=32, shuffle=True),
    "val_64": DataLoader(data["val"], batch_size=64, shuffle=True),
    "val_128": DataLoader(data["val"], batch_size=128, shuffle=True),
}


checkpoints = [
    "/train-classifier-32/",
    "/train-classifier-64/",
    "/train-classifier-128/",
]


for checkpoint_name in checkpoints:
    if "32" in checkpoint_name:
        print("32", checkpoint_name)
        train_dataloader = batch_dataloaders["train_32"]
        val_dataloader = batch_dataloaders["val_32"]
    elif "64" in checkpoint_name:
        print("64", checkpoint_name)
        train_dataloader = batch_dataloaders["train_64"]
        val_dataloader = batch_dataloaders["val_64"]

    elif "128" in checkpoint_name:
        print("128", checkpoint_name)
        train_dataloader = batch_dataloaders["train_128"]
        val_dataloader = batch_dataloaders["val_128"]

    epochs = 30

    model = VGG16FineTuneGeneric(get_model("classifier"))

    train_loss, train_acc, val_loss, val_acc = train(
        model,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_dir + checkpoint_name,
    )
