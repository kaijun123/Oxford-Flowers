import pandas as pd
import numpy as np
import scipy
from PIL import Image
import os

# PyTorch
from torchvision import transforms, datasets, models
from torchvision.transforms import v2
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn import metrics
import json
from timeit import default_timer as timer
from collections import OrderedDict
from torch.utils.data import default_collate
import matplotlib.pyplot as plt


class CustomException(Exception):
    pass


####################### Constants ##########################

current_directory = os.getcwd()
train_dir = current_directory + "/dataset/train"
test_dir = current_directory + "/dataset/test"
valid_dir = current_directory + "/dataset/valid"

checkpoint_dir = "checkpoint"
model_names = [
    "add-1-extra-layer-batch-32/",
    "add-1-extra-layer-batch-64/",
    "add-1-extra-layer-batch-128/",
    "add-1-extra-layer-batch-32-patience-5/",
    "add-1-extra-layer-batch-64-patience-5/",
    "add-1-extra-layer-batch-128-patience-5/",

    "add-1-extra-layer-with-new-augmentation-batch-32/",
    "add-1-extra-layer-with-new-augmentation-batch-64/",
    "add-1-extra-layer-with-new-augmentation-batch-128/",
    "add-1-extra-layer-with-augmentation-batch-64-patience-5/",

    "add-1-extra-layer-mixup-batch-32/",
    "add-1-extra-layer-mixup-batch-64/",
    "add-1-extra-layer-mixup-batch-128/",

    "train-classifier-batch-32/",
    "train-classifier-batch-64/",
    "train-classifier-batch-128/",
    "train-classifier-with-new-augmentation-batch-32/",
    "train-classifier-with-new-augmentation-batch-64/",
    "train-classifier-with-new-augmentation-batch-128/",

    "train-modified-batch-32/",
    "train-modified-batch-64/",
    "train-modified-batch-128/",
]

####################### Dataset and Dataloaders ##########################

weights = models.VGG16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

NUM_CLASSES = 102

mixup = v2.MixUp(num_classes=NUM_CLASSES)

# Image transformations
image_transforms = {
    # Train uses data augmentation
    "train_augment": transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=(-30, 30)),
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
    "test": preprocess,
}

data = {
    "train": datasets.ImageFolder(root=train_dir, transform=preprocess),
    "train_augment": datasets.ImageFolder(
        root=train_dir, transform=image_transforms["train_augment"]
    ),
    "val": datasets.ImageFolder(root=valid_dir, transform=preprocess),
}


def get_batch_dataloaders(data_type, batch_size, is_mix_up):
    if data_type not in ["train", "train_augment", "val", "test"] or batch_size not in [
        32,
        64,
        128,
    ]:
        raise CustomException("Not one of the acceptable inputs")

    def collate_fn(batch):
        return mixup(*default_collate(batch))

    if not is_mix_up:
        return DataLoader(data[data_type], batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(
            data[data_type], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )


def get_file_label_mapping(base_dir, data_split="test"):
    if data_split == "train":
        key = "trnid"
    elif data_split == "val":
        key = "valid"
    elif data_split == "test":
        key = "tstid"

    # print("key", key)

    # get the actual classification for the test set
    set_dict = scipy.io.loadmat(base_dir + "/setid.mat")

    label_idx_arr = []
    filename_arr = []
    for image_number in set_dict[key][0]:
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


####################### MODELS ##########################


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

    # finetune the last layer only
    if finetune == "last":
        last_input_feature = base.classifier[6].in_features  # Input to the last layer
        base.classifier = nn.Sequential(
            *list(base.classifier[:-1]),  # Retain all layers except the last one
            nn.Linear(last_input_feature, 102),  # Output layer for 102 classes
        )

        # Freeze all layers except the last layer of the classifier
        for param in base.features.parameters():
            param.requires_grad = False

        for param in base.classifier[:-1].parameters():
            param.requires_grad = False

    # finetune the classifier module
    elif finetune == "classifier":
        last_input_feature = base.classifier[6].in_features  # Input to the last layer
        base.classifier = nn.Sequential(
            *list(base.classifier[:-1]),  # Retain all layers except the last one
            nn.Linear(last_input_feature, 102),  # Output layer for 102 classes
        )

        # freeze feature module only
        for param in base.features.parameters():
            param.requires_grad = False

    # finetune the classifier module
    elif finetune == "modified":
        first_input_feature = base.classifier[
            0
        ].in_features  # Input to the first classifier layer
        base.classifier = nn.Sequential(
            nn.Linear(first_input_feature, 102),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(102, 102),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(102, 102),
        )

        # freeze feature module only
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


####################### Train ##########################


# ensure that the necessary layers in the model are frozen
def train(
    model, epochs, train_dataloader, val_dataloader, base_checkpoint_name, patience=None
):
    # print("base_checkpoint_name", base_checkpoint_name)
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(base_checkpoint_name, exist_ok=True)

    # initialise the optimizer and the criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # create the early stopper object
    if patience is not None:
        early_stopper = EarlyStopper(patience=patience)
    else:
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
            # print("X.shape", X.shape)
            if train_on_gpu:
                X, y_true = X.cuda(), y_true.cuda()

            # print("y_true", y_true[0])

            # Clear gradient
            optimizer.zero_grad()
            # Forward Pass
            y_pred = model(X)
            # print("y_pred", y_pred[0])

            # calculate the loss
            loss = criterion(y_pred, y_true)
            # print("loss", loss)
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

        print(
            "train_loss:",
            train_loss[-1],
            "val_loss:",
            val_loss[-1],
            "train_acc:",
            train_acc[-1],
            "val_acc:",
            val_acc[-1],
        )

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


def get_results(results_dir):
    filename = results_dir + "results.json"
    f = open(filename)
    data = json.load(f)

    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    for x in data.values():
        val_loss = x["val_loss"]
        val_acc = x["val_acc"]
        train_loss = x["train_loss"]
        train_acc = x["train_acc"]

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "train_acc": train_acc,
    }


####################### Test ##########################


def get_model_from_weights(model_type, model_weights_path):

    model = get_model(model_type)

    checkpoint = torch.load(model_weights_path)
    state_dict = checkpoint["model_state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("model.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model


def find_best_performing_model_checkpoint(model_name):
    if "add-1-extra-layer" in model_name:
        model_type = "last"
    elif "train-classifier" in model_name:
        model_type = "classifier"
    elif "train-modified" in model_name:
        model_type = "modified"

    data = get_results(checkpoint_dir + "/" + model_name)
    val_acc = data["val_acc"]
    max_epoch_minus_one = np.argmax(val_acc)

    weights_path = checkpoint_dir + "/" + model_name + f"epoch-{max_epoch_minus_one+1}.pt"
    print("model_name:", model_name, "max_epoch:", max_epoch_minus_one+1)

    return get_model_from_weights(model_type, weights_path)


def test(model, test_dataloader, model_name):
    print("model_name", model_name)

    # check if gpu is available. If yes, use gpu
    train_on_gpu = cuda.is_available()
    print("train_on_gpu:", train_on_gpu)
    if train_on_gpu:
        model = model.to("cuda")

    num_correct_predictions = 0
    total_samples = 0
    # test phase
    with torch.no_grad():
        # set the model to eval phase
        model.eval()

        for X, y_true in test_dataloader:
            print("X", X.shape)

            if train_on_gpu:
                X, y_true = X.cuda(), y_true.cuda()
            # calculate the loss
            y_pred = model(X)

            softmax = nn.Softmax(
                dim=1
            )  # Apply Softmax along the class dimension (dim=1)
            y_pred_probabilities = softmax(y_pred)
            # print("y_pred_probabilities", y_pred_probabilities)

            predicted_classes = torch.argmax(y_pred_probabilities, dim=1)
            # print("predicted_classes", predicted_classes)
            # print("y_true", y_true)

            # Convert to numpy for metrics
            y_pred_classes = predicted_classes.cpu().numpy()
            y_true = y_true.cpu().numpy()

            # Calculate the accuracy for this batch
            batch_accuracy = metrics.accuracy_score(y_true, y_pred_classes)

            num_correct_predictions += batch_accuracy * len(X)
            total_samples += len(X)
            print(
                "num_correct_predictions:",
                num_correct_predictions,
                "batch_samples:",
                len(X),
                "batch_accuracy:",
                batch_accuracy,
            )

    print("num_correct_predictions", num_correct_predictions)
    print("total_samples", total_samples)
    return num_correct_predictions / total_samples


####################### Predict ##########################


def predict(model_weights_path, image_path):
    if "add-1-extra-layer" in model_weights_path:
        model_type = "last"
    elif "train-classifier" in model_weights_path:
        model_type = "classifier"
    elif "train-modified" in model_weights_path:
        model_type = "modified"

    model = get_model_from_weights(model_type, model_weights_path)

    X = Image.open(image_path)
    X_transformed = preprocess(X)
    X_transformed = X_transformed.unsqueeze(0)
    y_pred = model.forward(X_transformed)
    # print(type(y_pred))

    softmax = nn.Softmax()
    category = np.argmax(softmax(y_pred).detach().numpy())

    # read the mapping from the json file
    file = open("cat_to_name.json")
    categoryToTypeMap = json.load(file)
    return category + 1, categoryToTypeMap[str(category + 1)]


####################### Visualisation ##########################
def visualise(models, metric):
    # get the results that you want
    collated_results = {
        model: get_results(f"{checkpoint_dir}/{model}")
        for model in models
    }

    for model, results in collated_results.items():
        y = results[metric]
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=model)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)

    plt.legend(bbox_to_anchor=(0.5, 0.5), fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def visualiseAll(models):
    # get the results that you want
    collated_results = {
        model: get_results(f"{checkpoint_dir}/{model}") for model in models
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    metrics = ["train_acc", "val_acc", "train_loss", "val_loss"]
    titles = [
        "Training Accuracy",
        "Validation Accuracy",
        "Training Loss",
        "Validation Loss",
    ]

    # print("axs", type(axs))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        print("ax", type(ax))
        for model, results in collated_results.items():
            if "add-1-extra-layer" in model:
                label = model.replace("add-1-extra-layer", "method-1")
            elif "train-classifier" in model:
                label = model.replace("train-classifier", "method-2")
            elif "train-modified" in model:
                label = model.replace("train-modified", "method-3")

            y = results[metrics[i]]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metrics[i])
        ax.set_title(titles[i])

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("This is the utils.py file.")
