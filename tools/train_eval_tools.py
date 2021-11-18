import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from tools.misc_tools import create_path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix


########################################################################################################################
#
# Training operation
#
########################################################################################################################


def train_op(
    device,
    model,
    train_loader,
    val_loader,
    output_dim,
    optimizer,
    loss_function,
    epochs,
    summary_writer_obj=None,
):

    """

        Training operation for a single model.

        :param device: Device.
        :param model: Model.
        :param train_loader: Train data loader.
        :param val_loader: Validation data loader.
        :param output_dim: Output model dimensions.
        :param optimizer: Optimizer.
        :param loss_function: Loss function.
        :param epochs: Number of training epochs.
        :param summary_writer_obj: Summary writer object.
        :return: The train loss.

    """

    # Running loss and total samples init.
    running_loss, samples = 0.0, 0

    # Output dictionary.
    output_dict = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Loop over epochs.
    # ------------------------------------------------------------------------------------------------------------------

    # Progress bar.
    pbar = tqdm(range(epochs))

    for ep in pbar:

        # Enable model training.
        model.train()

        # --------------------------------------------------------------------------------------------------------------
        # For the current epoch, loop over samples in the data loader.
        # --------------------------------------------------------------------------------------------------------------

        for x, y in train_loader:

            # Get data samples.
            x, y = x.to(device), y.to(device)

            # Clean the optimizer.
            optimizer.zero_grad()

            # Compute the loss.
            loss = 0.0
            if isinstance(loss_function, nn.CrossEntropyLoss):
                loss = loss_function(model(x), y)
            elif isinstance(loss_function, nn.BCEWithLogitsLoss):
                loss = loss_function(model(x), torch.unsqueeze(y.float(), 1))

            # Running loss.
            running_loss += loss.item() * y.shape[0]

            # Count data samples.
            samples += y.shape[0]

            # Compute gradients.
            loss.backward()

            # Single training step.
            optimizer.step()

        # --------------------------------------------------------------------------------------------------------------
        # Performance metric.
        # --------------------------------------------------------------------------------------------------------------

        train_loss = running_loss / samples
        output_dict = {"train/loss": [train_loss]}

        # --------------------------------------------------------------------------------------------------------------
        # Summary writer (Tensorboard).
        # --------------------------------------------------------------------------------------------------------------

        if summary_writer_obj is not None:

            for key, value in output_dict.items():

                summary_writer_obj.add_scalar(
                    tag=key,
                    scalar_value=value[0],
                    global_step=ep
                )

                summary_writer_obj.flush()

        # --------------------------------------------------------------------------------------------------------------
        # Progress bar message.
        # --------------------------------------------------------------------------------------------------------------

        pbar.set_description("[ Epoch = {}/{} ] Train loss = {:0.4f}".format(ep + 1, epochs, train_loss))

        # --------------------------------------------------------------------------------------------------------------
        # Model evaluation (validation set)
        # --------------------------------------------------------------------------------------------------------------

        # Evaluation operation.
        _ = eval_op(
            device=device,
            model=model,
            loader=val_loader,
            output_dim=output_dim,
            bc_threshold=0.5,
            dataset_split='val',
            summary_writer_obj=summary_writer_obj,
            global_step=ep,
        )

    return output_dict


########################################################################################################################
#
# Evaluation operation
#
########################################################################################################################


def eval_op(
    device,
    model,
    loader,
    output_dim=2,
    bc_threshold=0.5,
    dataset_split='val',
    summary_writer_obj=None,
    global_step=0
):

    """
        Evaluation operation for a single model.

        :param device: Device.
        :param model: Model.
        :param loader: Data loader.
        :param output_dim: Number of output model dimensions.
        :param bc_threshold: Binary classification threshold (valid when output_dim = 1). Default: 0.5.
        :param dataset_split: Dataset split. Use: "train", "val", "test".
        :param summary_writer_obj: Summary writer object.
        :param global_step: Global step (useful for tensorboard visualization).
        :return: The accuracy of the model on the specified dataset split.

    """

    # Set the current model in evaluation mode.
    model.eval()

    # Initialize total and correct samples.
    samples, correct = 0, 0

    # Variables to hold all ground-truth and predicted classes.
    y_true_all = None
    y_pred_all = None
    y_score_all = None

    with torch.no_grad():

        # --------------------------------------------------------------------------------------------------------------
        # Loop over samples in the data loader.
        # --------------------------------------------------------------------------------------------------------------

        for i, (x, y) in enumerate(loader):

            # Get data samples.
            x, y = x.to(device), y.to(device)

            if output_dim == 2:

                # Forwards pass.
                y_ = F.softmax(model(x), dim=1)

                # Predicted class.
                _, predicted = torch.max(y_.detach(), 1)

            elif output_dim == 1:

                # Reshape ground-truth.
                y = torch.unsqueeze(y.float(), 1)

                # Forwards pass.
                y_ = torch.sigmoid(model(x))

                # Predicted class.
                predicted = y_.detach()
                predicted[predicted >= bc_threshold] = 1
                predicted[predicted < bc_threshold] = 0

            # Count samples.
            samples += y.shape[0]

            # Count correct samples.
            correct += (predicted == y).sum().item()

            # Concatenate ground-truth and predictions in separated vectors.
            if i == 0:

                # Predicted scores.
                y_score_all = y_.detach().cpu().numpy()

                # Predicted class.
                y_pred_all = predicted.cpu().numpy()

                # Ground-truth class.
                y_true_all = y.cpu().numpy()

            else:

                # Predicted scores.
                y_score_all = np.concatenate((y_score_all, y_.detach().cpu().numpy()), axis=0)

                # Predicted class.
                y_pred_all = np.concatenate((y_pred_all, predicted.cpu().numpy()), axis=0)

                # Ground-truth class.
                y_true_all = np.concatenate((y_true_all, y.cpu().numpy()), axis=0)

        # --------------------------------------------------------------------------------------------------------------
        # Performance metrics.
        # --------------------------------------------------------------------------------------------------------------

        # Accuracy
        total_accuracy = correct/samples

        # Precision, recall, f1-score.
        total_precision, total_recall, total_f1_score, _ = precision_recall_fscore_support(
            y_true_all,
            y_pred_all,
            beta=1.0,
            pos_label=1,
            average='binary'
        )

        # Total ROC AUC
        if output_dim == 2:
            total_roc_auc = roc_auc_score(y_true_all, y_score_all[:, 1])
        elif output_dim == 1:
            total_roc_auc = roc_auc_score(y_true_all, y_score_all)

        # TN, FP, FN, TP.
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all).ravel()

        # Specificity.
        specificity = tn / (tn + fp)

        # Output dict.
        output_dict = {
            "{}/accuracy".format(dataset_split): [total_accuracy],
            "{}/precision".format(dataset_split): [total_precision],
            "{}/recall".format(dataset_split): [total_recall],
            "{}/f1_score".format(dataset_split): [total_f1_score],
            "{}/roc_auc".format(dataset_split): [total_roc_auc],
            "{}/specificity".format(dataset_split): [specificity]
        }

        # --------------------------------------------------------------------------------------------------------------
        # Summary writer (Tensorboard).
        # --------------------------------------------------------------------------------------------------------------

        if summary_writer_obj is not None:

            for key, value in output_dict.items():

                summary_writer_obj.add_scalar(
                    tag=key,
                    scalar_value=value[0],
                    global_step=global_step
                )

                summary_writer_obj.flush()

    return output_dict

########################################################################################################################
#
# Compute model size.
#
########################################################################################################################


def compute_model_size(model, log_path=None, model_name=None, summary_writer_obj=None, global_step=0):

    """

    Computes the model size.

    :param model:
    :param log_path:
    :param model_name:
    :param summary_writer_obj:
    :param global_step:
    :return:
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the total number of parameters and model size in MB.
    # ------------------------------------------------------------------------------------------------------------------

    # Total parameters.
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size in MB.
    model_size_in_mb = 32. * total_parameters / (8. * 1e6)

    # Output dict.
    output_dict = {
        "model_size/total_parameters": [total_parameters],
        "model_size/size_in_mb": [model_size_in_mb],
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Summary writer (Tensorboard).
    # ------------------------------------------------------------------------------------------------------------------

    if summary_writer_obj is not None:

        for key, value in output_dict.items():
            summary_writer_obj.add_scalar(
                tag=key,
                scalar_value=value[0],
                global_step=global_step
            )

            summary_writer_obj.flush()

    # ------------------------------------------------------------------------------------------------------------------
    # Save data on disk (CSV file).
    # ------------------------------------------------------------------------------------------------------------------

    if (log_path is not None) and (model_name is not None):
        create_path(path=log_path)
        output_csv_ffname = '{}/{}_model_size.csv'.format(log_path, model_name)
        output_df = pd.DataFrame(output_dict)
        output_df.to_csv(output_csv_ffname)

    return output_dict
