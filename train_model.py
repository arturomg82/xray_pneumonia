"""

This project shows the concept of transfer learning, using a variety of pre-trained convolutional neural networks
(e.g., Resnet-18, Resnet-50, Mobilenet-V2, Mobilenet-V3) in the ImageNet dataset. The target task is the classification
of normal vs. abnormal x-ray images, where an abnormal image corresponds to a case of pneumonia.

The dataset used to test this code is publicly available in:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

"""

import click
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tools.misc_tools import get_data_loaders, create_path
from tools.train_eval_tools import train_op, eval_op, compute_model_size
from tools.load_models_tools import get_model, get_model_parameters


@click.command()
@click.argument('model_name')
@click.option(
    '--dataset-path',
    type=click.Path(),
    default=None,
    metavar='TRAIN_DATASET_PATH',
    help='Path to the training dataset.'
)
@click.option(
    '--log-path',
    type=click.Path(),
    default=None,
    metavar='LOG_PATH',
    help='Path to save data (e.g., experimental results).'
)
@click.option(
    '--num-classes',
    type=int,
    default=2,
    metavar='NUM_CLASSES',
    help='Number of classes.'
)
@click.option(
    '--loss-function-type',
    type=str,
    default="CrossEntropyLoss",
    metavar='LOSS_FUNCTION_TYPE',
    help='Loss function type: "BinaryCrossEntropyLoss", "CrossEntropyLoss".'
)
@click.option(
    '--train-batch-size',
    type=int,
    default=64,
    metavar='TRAIN_BATCH_SIZE',
    help='Train batch size.'
)
@click.option(
    '--val-batch-size',
    type=int,
    default=128,
    metavar='VAL_BATCH_SIZE',
    help='Validation batch size.'
)
@click.option(
    '--epochs',
    type=int,
    default=10,
    metavar='EPOCHS',
    help='Total training epochs.'
)
@click.option(
    '--learning-rate',
    type=float,
    default=0.0001,
    metavar='LR',
    help='Learning rate.'
)
@click.option(
    '--seed-number',
    type=int,
    default=20082021,
    metavar='SEED_NUMBER',
    help='Seed number.'
)
@click.option(
    '--enable-training',
    is_flag=True,
    default=False,
    help='If True, model training will be carried out.'
)
@click.option(
    '--train-last-layer',
    is_flag=True,
    default=False,
    help='If True, only the last layer for the neural network will be optimized.'
)
@click.option(
    '--verbose',
    is_flag=True,
    default=False,
    help='If True, messages will be displayed.'
)
def main(
    model_name,
    dataset_path,
    log_path,
    num_classes,
    loss_function_type,
    train_batch_size,
    val_batch_size,
    epochs,
    learning_rate,
    seed_number,
    enable_training,
    train_last_layer,
    verbose
):

    print('[ Chest X-Ray Images (Pneumonia) ][ Binary Classification (Normal vs. Pneumonia) ]\n')

    # ------------------------------------------------------------------------------------------------------------------
    # Initialization.
    # ------------------------------------------------------------------------------------------------------------------

    # Seed numbers for reproducibility.
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    # Device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print('[ Device = {} ]\n'.format(device))

    # Create main output path.
    create_path(path=log_path)

    # Create train and validation data loaders.
    train_loader, val_loader, test_loader, train_dataset_properties, _, _ = get_data_loaders(
        dataset_path=dataset_path,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=val_batch_size,
        train_workers=6,
        val_workers=4,
        test_workers=4,
        verbose=verbose,
    )

    # Create a summary writer object.
    tensorboard_path = "{}/tensorboard".format(log_path)
    create_path(path=tensorboard_path)
    summary_writer_obj = SummaryWriter(log_dir=tensorboard_path)

    if verbose:
        print('[ Summary writer object created. ]')
        print(' ')

    # Output model dimensions.
    output_model_dimensions =\
        num_classes - 1 if (num_classes == 2) and (loss_function_type == "BinaryCrossEntropyLoss") else num_classes

    # Model.
    model = get_model(
        device=device,
        model_name=model_name,
        output_dim=output_model_dimensions,
        verbose=verbose
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Model training and evaluation.
    # ------------------------------------------------------------------------------------------------------------------

    if enable_training:

        # --------------------------------------------------------------------------------------------------------------
        # Compute model size.
        # --------------------------------------------------------------------------------------------------------------

        _ = compute_model_size(
            model=model,
            log_path=log_path,
            model_name=model_name,
            summary_writer_obj=summary_writer_obj,
            global_step=0
        )

        # --------------------------------------------------------------------------------------------------------------
        # Optimizer and model parameters.
        # --------------------------------------------------------------------------------------------------------------

        # Select the model parameters to be optimized.
        model_parameters = get_model_parameters(
            model_name=model_name,
            model=model,
            last_layer=train_last_layer,
            verbose=verbose
        )

        # Optimizer.
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)

        # --------------------------------------------------------------------------------------------------------------
        # Compute a vector of weights per class to counteract the class imbalance in the dataset.
        # --------------------------------------------------------------------------------------------------------------

        weight_vector = torch.Tensor(np.array(train_dataset_properties['train/total_samples_per_class'])).to(device)
        weight_vector /= weight_vector.sum()
        weight_vector = 1. - weight_vector

        # --------------------------------------------------------------------------------------------------------------
        # Loss function definition.
        # --------------------------------------------------------------------------------------------------------------

        assert loss_function_type in ["BinaryCrossEntropyLoss", "CrossEntropyLoss"], \
            "[ Error ] Loss function not implemented: {}".format(loss_function_type)

        loss_function = None
        if num_classes >= 2 and loss_function_type == "CrossEntropyLoss":

            loss_function = nn.CrossEntropyLoss(weight=weight_vector)

            if verbose:
                print('[ Loss function = {} ]'.format(loss_function_type))
                print('\t- Number of classes = {}'.format(num_classes))
                print('\t- Samples per class = {}'.format(train_dataset_properties['train/total_samples_per_class']))
                print('\t- Weight vector = {}'.format(weight_vector))
                print(' ')

        elif num_classes == 2 and loss_function_type == "BinaryCrossEntropyLoss":

            loss_function = nn.BCEWithLogitsLoss(pos_weight=weight_vector[1])

            if verbose:
                print('[ Loss function = {} ]'.format(loss_function_type))
                print('\t- Number of classes = {}'.format(num_classes))
                print('\t- Samples per class = {}'.format(train_dataset_properties['train/total_samples_per_class']))
                print('\t- Weight vector = {}'.format(weight_vector))
                print(' ')

        if verbose:
            print('[ Model training... ]\n')

        # --------------------------------------------------------------------------------------------------------------
        # Model training.
        # --------------------------------------------------------------------------------------------------------------

        train_stats = train_op(
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dim=output_model_dimensions,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=epochs,
            summary_writer_obj=summary_writer_obj,
        )

        if verbose:
            print('\n[ Training stage completed ]')
            for key, value in train_stats.items():
                print('\t[ Model = {} ] {} = {}'.format(model_name, key, value[0]))
            print(' ')

        if verbose:
            print('[ Model evaluation (validation set) ... ]\n')

        # --------------------------------------------------------------------------------------------------------------
        # Model evaluation (test set).
        # --------------------------------------------------------------------------------------------------------------

        # Evaluation operation.
        test_stats = eval_op(
            device=device,
            model=model,
            loader=test_loader,
            output_dim=output_model_dimensions,
            bc_threshold=0.5,
            dataset_split='test',
            summary_writer_obj=summary_writer_obj,
            global_step=0,
        )

        if verbose:
            print('[ Evaluation stage completed (test set) ]')
            for key, value in test_stats.items():
                print('\t[ Model = {} ] {} = {}'.format(model_name, key, value[0]))
            print(' ')

        # Write model performance on disk (CSV file).
        test_csv_ffname = '{}/{}_test_performance.csv'.format(log_path, model_name)
        test_stats_df = pd.DataFrame(test_stats)
        test_stats_df.to_csv(test_csv_ffname)

        if verbose:
            print('[ Model performance (test set) ] Saved in: {} '.format(test_csv_ffname))
            print(' ')

        # --------------------------------------------------------------------------------------------------------------
        # Save model on disk.
        # --------------------------------------------------------------------------------------------------------------

        model_ffname = "{:s}/{:s}.pt".format(log_path, model_name)
        torch.save({'model_state_dict': model.state_dict()}, model_ffname)
        if verbose:
            print('[ Model saved on disk ]')
            print('\t- File: {}'.format(model_ffname))
            print(' ')

    # Close summary writer object.
    summary_writer_obj.close()


if __name__ == '__main__':
    main()
