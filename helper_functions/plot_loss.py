"""
Stores a function that plots the train and loss functions of a model
"""
import matplotlib.pyplot as plt
from helper_functions.result_types import TrainTestResult


def plot_loss(model_details: TrainTestResult) -> None:
    """
    Plots the train and test losses and accuracy of the model.

    :param model_details: A `TrainTestResult` of the model whose
        losses and accuracies you want to plot.
    :return: Returns nothing
    """
    model_train_loss, model_train_acc = [], []
    model_test_loss, model_test_acc = [], []

    for train_details, test_details in zip(model_details['train'], model_details['test']):
        model_train_loss.append(train_details['loss'].detach().cpu())
        model_train_acc.append(train_details['acc'])

        model_test_loss.append(test_details['loss'].detach().cpu())
        model_test_acc.append(test_details['acc'])

    x_vals = list(range(0, len(model_train_loss)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 8))

    ax0.plot(x_vals, model_train_loss, label="Train")
    ax0.plot(x_vals, model_test_loss, label="Test")
    ax0.set_title("Loss")

    ax1.plot(x_vals, model_train_acc, label="Train")
    ax1.plot(x_vals, model_test_acc, label="Test")
    ax1.set_title("Acc")

    plt.legend()
