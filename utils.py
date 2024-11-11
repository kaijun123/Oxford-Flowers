import pandas as pd
import numpy as np
import scipy
from torch.utils.data import DataLoader, Dataset
from PIL import Image


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


if __name__ == "__main__":
    print("This is the module.py file.")
