import os
import glob
import torch
from torchvision import transforms, datasets


def create_path(path):

    """ Create path if it doesn't exist. """

    if not os.path.exists(path):
        os.makedirs(path)


def get_chest_xray_dataset(path, transform, data_split='train', verbose=True):

    """
    Returns a dataset object, for the specified split, corresponding to the the Chest X-Ray Images (Pneumonia) dataset,
    including its properties.

    """

    # Dataset path.
    dataset_path = "{:s}/{:s}".format(path, data_split)

    # Generic data loader for images.
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Dictionary to store dataset properties.
    dataset_properties = dict()
    dataset_properties['{}/class_names'.format(data_split)] = dataset.classes
    dataset_properties['{}/total_samples_per_class'.format(data_split)] = [0] * len(dataset.classes)
    dataset_properties['{}/total_samples'.format(data_split)] = len(dataset)

    # Get the number of samples per class.
    for i, class_name in enumerate(dataset_properties['{}/class_names'.format(data_split)]):
        dataset_properties['{}/total_samples_per_class'.format(data_split)][i] = \
            len(glob.glob('{}/{}/*.jpeg'.format(dataset_path, class_name)))

    if verbose:
        print('[ X-Ray Dataset | Split {} ]'.format(data_split))
        print('\t- Dataset path = {}'.format(dataset_path))
        print('\t- Class names = {}'.format(dataset_properties['{}/class_names'.format(data_split)]))
        print('\t- Total samples per class = {}'.format(dataset_properties['{}/total_samples_per_class'.format(data_split)]))
        print('\t- Total samples = {}'.format(dataset_properties['{}/total_samples'.format(data_split)]))
        print(' ')

    return dataset, dataset_properties


def get_data_loaders(
    dataset_path,
    train_batch_size=64,
    val_batch_size=32,
    test_batch_size=32,
    train_workers=6,
    val_workers=4,
    test_workers=4,
    verbose=True
):

    """ Create data loaders for the train, validation, and test sets. """

    # ------------------------------------------------------------------------------------------------------------------
    # Train loader
    # ------------------------------------------------------------------------------------------------------------------

    train_dataset_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset, train_dataset_properties = \
        get_chest_xray_dataset(
            path=dataset_path,
            transform=train_dataset_transform,
            data_split='train',
            verbose=verbose
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_workers,
    )

    if verbose:
        print('[ Train data loader created. ]')
        print('\t- Batch size = {}'.format(train_batch_size))
        print('\t- Number of workers = {}'.format(train_workers))
        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Validation loader.
    # ------------------------------------------------------------------------------------------------------------------

    val_dataset_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    val_dataset, val_dataset_properties = \
        get_chest_xray_dataset(
            path=dataset_path,
            transform=val_dataset_transform,
            data_split='val',
            verbose=verbose
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=True,
        num_workers=val_workers,
    )

    if verbose:
        print('[ Validation data loader created. ]')
        print('\t- Batch size = {}'.format(val_batch_size))
        print('\t- Number of workers = {}'.format(val_workers))
        print(' ')

    # ------------------------------------------------------------------------------------------------------------------
    # Test loader.
    # ------------------------------------------------------------------------------------------------------------------

    test_dataset_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_dataset, test_dataset_properties = \
        get_chest_xray_dataset(
            path=dataset_path,
            transform=test_dataset_transform,
            data_split='test',
            verbose=verbose
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=test_workers,
    )

    if verbose:
        print('[ Test data loader created. ]')
        print('\t- Batch size = {}'.format(val_batch_size))
        print('\t- Number of workers = {}'.format(val_workers))
        print(' ')

    return train_loader, val_loader, test_loader, train_dataset_properties, val_dataset_properties, test_dataset_properties
