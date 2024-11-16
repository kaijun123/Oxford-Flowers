# SC4001 Neural Network Project

### Introduction
- In this repo, you will find code for transfer learning of vgg16 for the Oxford Flowers 102 dataset
- 3 main architectures used:
  - Models with the prefix "add-1-extra-layer": have the last FC layer replaced with one that ouputs 102 features. Only the last FC is trained
  - Models with the prefix "train-classifier": also have the last FC replaced, but the whole classifier module is finetuned
  - Models with the prefix "train-modified": have the classifier module modified to have fewer neurones in the hidden layers. The whole classifier module is also trained from scratch

### Repo layout
- `utils.py`: contains utility code
- `train.py`: contains code that is used to train the model
- `visualisation.ipynb`: is used to visualise the train and validation results
- `cnn.ipynb`: contains the main bulk of the test code


### Download the dataset
- Run the following command in the terminal:
```
curl -L -o ./archive.zip https://www.kaggle.com/api/v1/datasets/download/nunenuh/pytorch-challange-flower-dataset
```
- Run this cell in `cnn.ipynb`
```
# install the data using torch, to get the full dataset, especially for the test set
dataset = datasets.Flowers102(root=current_directory, download=True)
dataset
```
- Download the weights for the model at `https://huggingface.co/kaijun123/sc4001-oxford-102` and save it in the `checkpoint` directory
- We are using a combination of both methods. The data returned using the curl sorts the training and validation set into folders which represent their actual labels, which makes it convenient for training and validation. But it is missing some test data, hence, we will need to get the remaining from torch


