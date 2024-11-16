from utils import (
    train,
    get_batch_dataloaders,
    checkpoint_dir,
    VGG16FineTuneGeneric,
    get_model,
)

# (checkpoint_name, train_dataset_type, batch_size, model_type, patience)
parameters = [
    # replace last layer
    ("/add-1-extra-layer-batch-32/", "train", 32, "last", 3, False),
    ("/add-1-extra-layer-batch-64/", "train", 64, "last", 3, False),
    ("/add-1-extra-layer-batch-128/", "train", 128, "last", 3, False),
    ("/add-1-extra-layer-mixup-batch-32/", "train", 32, "last", 3, True),
    ("/add-1-extra-layer-mixup-batch-64/", "train", 64, "last", 3, True),
    ("/add-1-extra-layer-mixup-batch-128/", "train", 128, "last", 3, True),
    ("/add-1-extra-layer-with-augmentation-batch-64/", "train", 64, "last", 3, False),
    ("/add-1-extra-layer-with-augmentation-batch-32/", "train", 32, "last", 3, False),
    ("/add-1-extra-layer-with-augmentation-batch-128/", "train", 128, "last", 3, False),
    ("/add-1-extra-layer-batch-32-patience-5/", "train", 32, "last", 5, False),
    ("/add-1-extra-layer-batch-64-patience-5/", "train", 32, "last", 5, False),
    ("/add-1-extra-layer-batch-128-patience-5/", "train", 32, "last", 5, False),
    ("add-1-extra-layer-with-augmentation-batch-64-patience-5/", "train_augment", 32, "last", 5, False),
    # finetune the entire classifer module
    ("/train-classifier-32/", "train", 32, "classifier", 3, False),
    ("/train-classifier-64/", "train", 64, "classifier", 3, False),
    ("/train-classifier-128/", "train", 128, "classifier", 3, False),
    ("/train-classifier-with-augmentation-32/", "train_augment", 32, "classifier", 3, False),
    ("/train-classifier-with-augmentation-64/", "train_augment", 64, "classifier", 3, False),
    ("/train-classifier-with-augmentation-128/", "train_augment", 128, "classifier", 3, False),
    # modify the architecture of classifer module
    ("/train-modified-batch-32/", "train", 32, "modified", 3, False),
    ("/train-modified-batch-64/", "train", 64, "modified", 3, False),
    ("/train-modified-batch-128/", "train", 128, "modified", 3, False),
]

for p in parameters:
    checkpoint_name, train_dataset_type, batch_size, model_type, patience, is_mix_up = p

    model = VGG16FineTuneGeneric(get_model(model_type))
    train_dataloader = get_batch_dataloaders(train_dataset_type, batch_size, is_mix_up)
    val_dataloader = get_batch_dataloaders("val", batch_size, False)
    print("model", model)

    epochs = 30
    train_loss, train_acc, val_loss, val_acc = train(
        model,
        epochs,
        train_dataloader,
        val_dataloader,
        checkpoint_dir + checkpoint_name,
        patience=patience,
    )
