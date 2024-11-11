import pandas as pd
import numpy as np
import scipy
from PIL import Image
import os

# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn import metrics
import json
from timeit import default_timer as timer


# Dataset and Dataloaders

weights = models.VGG16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

current_directory = os.getcwd()
train_dir = current_directory + "/dataset/train"
test_dir = current_directory + "/dataset/test"
valid_dir = current_directory + "/dataset/valid"

checkpoint_dir = "checkpoint"

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
}

data = {
    "train": datasets.ImageFolder(root=train_dir, transform=preprocess),
    "train_augment": datasets.ImageFolder(
        root=train_dir, transform=image_transforms["train_augment"]
    ),
    "val": datasets.ImageFolder(root=valid_dir, transform=preprocess),
}


def get_file_label_mapping(base_dir):
    # get the actual classification for the test set
    set_dict = scipy.io.loadmat(base_dir + "/setid.mat")

    label_idx_arr = []
    filename_arr = []
    for image_number in set_dict["tstid"][0]:
        label_idx_arr.append(image_number)
        image_number_str = str(image_number)
        filename = (
            "image_" + (5 - len(str(image_number))) * "0" + image_number_str + ".jpg"
        )
        filename_arr.append(filename)

    file_df = pd.DataFrame(data={"label_idx": label_idx_arr, "filename": filename_arr})

    mat_dict = scipy.io.loadmat(base_dir + "/imagelabels.mat")
    label_df = pd.DataFrame(
        data={"label_idx": np.arange(1, 8190), "labels": mat_dict["labels"][0]},
    )

    file_label_df = label_df.merge(file_df)
    return file_label_df


class TestImageDataset(Dataset):
    r"""
    Args:
        img_filename: an iterable of filenames to the image
        labels: an iterable of labels
        transform: transformations to the image
        target_transform: transformations to the label
    """

    def __init__(
        self, img_dir, img_filenames, labels, transform=None, target_transform=None
    ):
        self.img_dir = img_dir
        self.img_filenames = img_filenames
        self.img_labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_filenames[idx]
        label = self.img_labels[idx]

        # open the image
        path = self.img_dir + "/" + filename
        img = Image.open(path)

        # transform the image
        if self.transform is not None:
            img = self.transform(img)

        # transform the labels
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label



# MODELS


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_model(finetune):
    weights = models.VGG16_Weights.IMAGENET1K_V1
    base = models.vgg16(weights=weights)

    if finetune == "last":
        last_input_feature = base.classifier[6].in_features  # Input to the last layer
        base.classifier = nn.Sequential(
            *list(base.classifier[:-1]),  # Retain all layers except the last one
            nn.Linear(last_input_feature, 102),  # Output layer for 102 classes
        )

        # freeze all of the previous layers
        for param in base.parameters():
            param.requires_grad = False

    elif finetune == "classifier":
        last_input_feature = base.classifier[6].in_features  # Input to the last layer
        base.classifier = nn.Sequential(
            *list(base.classifier[:-1]),  # Retain all layers except the last one
            nn.Linear(last_input_feature, 102),  # Output layer for 102 classes
        )

        # freeze feature modules
        for param in base.features.parameters():
            param.requires_grad = False

    return base


class VGG16FineTuneGeneric(nn.Module):
    def __init__(self, model):
        super(VGG16FineTuneGeneric, self).__init__()

        self.model = model

        # VGG16's original feature extractor remains unchanged
        self.features = model.features
        self.classifier = model.classifier

    def forward(self, X):
        X = self.features(X)
        # flatten the feature maps into single vectors for fully connected layers
        X = torch.flatten(X, 1)
        y = self.classifier(X)
        return y


class VGG16FineTune(nn.Module):
    def __init__(self):
        super(VGG16FineTune, self).__init__()

        # get the base model
        # initiate new instance, so that they don't interfere with each other
        weights = models.VGG16_Weights.IMAGENET1K_V1
        base = models.vgg16(weights=weights)

        # freeze all of the previous layers
        for param in base.parameters():
            param.requires_grad = False

        # VGG16's original feature extractor remains unchanged
        self.features = base.features

        last_input_feature = base.classifier[6].in_features  # Input to the last layer
        self.classifier = nn.Sequential(
            *list(base.classifier[:-1]),  # Retain all layers except the last one
            nn.Linear(last_input_feature, 102),  # Output layer for 102 classes
        )

    def forward(self, X):
        X = self.features(X)
        # flatten the feature maps into single vectors for fully connected layers
        X = torch.flatten(X, 1)
        y = self.classifier(X)
        return y


# ensure that the necessary layers in the model are frozen
def train(model, epochs, train_dataloader, val_dataloader, base_checkpoint_name):
    print("base_checkpoint_name", base_checkpoint_name)
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(base_checkpoint_name, exist_ok=True)

    # initialise the optimizer and the criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # create the early stopper object
    early_stopper = EarlyStopper()

    # check if gpu is available. If yes, use gpu
    train_on_gpu = cuda.is_available()
    print("train_on_gpu:", train_on_gpu)
    if train_on_gpu:
        model = model.to("cuda")

    # store the loss, and accuracy across each epoch
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # train for the specific number of epochs
    for e in range(epochs):
        print(f"------ Start epoch: {e} ------")
        # find the average training loss and accuracy
        epoch_train_loss = []
        epoch_train_acc = []

        epoch_val_loss = []
        epoch_val_acc = []

        # train phase
        # set the mode of the model
        model.train()
        start = timer()
        for batch, (X, y_true) in enumerate(train_dataloader):

            # iterate over each batch using the train data loader

            if train_on_gpu:
                X, y_true = X.cuda(), y_true.cuda()

            # Clear gradient
            optimizer.zero_grad()
            # Forward Pass
            y_pred = model(X)

            # calculate the loss
            loss = criterion(y_pred, y_true)
            loss.backward()

            if train_on_gpu:
                epoch_train_loss.append(loss.detach().cpu().numpy())
            else:
                epoch_train_loss.append(loss.detach().numpy())

            # Update the parameters
            optimizer.step()

            # calculate the accuracy
            batch_accuracy = metrics.accuracy_score(
                y_true=y_true.cpu().numpy(),
                y_pred=y_pred.argmax(dim=1).detach().cpu().numpy(),
            )

            epoch_train_acc.append(batch_accuracy)

            # Track training progress
            print(
                f"Epoch: {e}\t{100 * (batch + 1) / len(train_dataloader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.",
                end="\r",
            )

        # validation phase
        with torch.no_grad():
            # set the model to eval phase
            model.eval()

            for X, y_true in val_dataloader:
                # iterate over each batch using the train data loader

                if train_on_gpu:
                    X, y_true = X.cuda(), y_true.cuda()

                # calculate the loss
                y_pred = model(X)
                loss = criterion(y_pred, y_true)

                if train_on_gpu:
                    epoch_val_loss.append(loss.detach().cpu().numpy())
                else:
                    epoch_val_loss.append(loss.detach().cpu().numpy())

                # calculate the accuracy
                batch_accuracy = metrics.accuracy_score(
                    y_true=y_true.cpu().numpy(),
                    y_pred=y_pred.argmax(dim=1).detach().cpu().numpy(),
                )

                epoch_val_acc.append(batch_accuracy)

        train_loss.append(np.mean(epoch_train_loss))
        val_loss.append(np.mean(epoch_val_loss))

        train_acc.append(np.mean(epoch_train_acc))
        val_acc.append(np.mean(epoch_val_acc))

        # early stopping
        if early_stopper.early_stop(val_loss[-1]):
            break

        # save the model parameters to a file
        checkpoint_name = base_checkpoint_name + f"epoch-{e+1}.pt"
        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            checkpoint_name,
        )

    results = {
        checkpoint_name: {
            "train_loss": [float(e) for e in train_loss],
            "train_acc": [float(e) for e in train_acc],
            "val_loss": [float(e) for e in val_loss],
            "val_acc": [float(e) for e in val_acc],
        }
    }

    # Write to JSON file to store the results
    with open(base_checkpoint_name + "results.json", "a") as f:
        json.dump(results, f, indent=4)  # `indent=4` makes the JSON pretty-printed

    return train_loss, train_acc, val_loss, val_acc


if __name__ == "__main__":
    print("This is the utils.py file.")
