"""
Provides a function for evaluating a model's predictions
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(
        model: nn.Module,
        dataloader: DataLoader,
        labels: List[str],
        device: torch.device = torch.device("cpu"),
) -> ConfusionMatrixDisplay:
    """
    This creates a plot of a confusion matrix for a model.
    To show the plot, use the following code:

    ```python
    conf_matrix = plot_confusion_matrix(model, dataloader, labels)
    conf_matrix.plot(cmap='gray', xticks_rotation='vertical')
    ```

    :param model: The model for the confusion_matrix
    :param dataloader: A dataloader containing validation
        or training data
    :param labels: Human readable labels
    :param device: The device where the script will run.
        Default is "cpu"
    :return: A `ConfusionMatrixDisplay`
    """

    y_true, y_pred = [], []

    model.to(device=device)
    model.eval()

    for image, label in dataloader:
        image, label = image.to(device=device), label.to(device=device)

        with torch.inference_mode():
            preds_logits = model(image)
            preds = preds_logits.argmax(dim=1)

            y_pred = [*y_pred, *preds.cpu()]
            y_true = [*y_true, *label.cpu()]

    y_pred = [labels[pred] for pred in y_pred]
    y_true = [labels[true] for true in y_true]

    return ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
        ),
        display_labels=labels,
    )
