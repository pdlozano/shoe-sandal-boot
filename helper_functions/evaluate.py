"""
Provides a function for evaluating a model on a given dataset
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List


def eval_model(
        model: nn.Module,
        dataloader: DataLoader,
        real_labels: List[str],
        device: torch.device = torch.device("cpu"),
) -> None:
    """
    Evaluates the first iteration of the dataloader.
    If you want to evaluate a part of the dataloader, consider
    using the following script:

    ```python
    new_dataloader = DataLoader(
        dataset=old_dataloader.dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    ```

    :param model: The model to be evaluated
    :param dataloader: A dataloader of the dataset
    :param real_labels: A list of human readable labels
    :param device: The device of your choosing. Default is 'cpu'
    :return: Returns nothing
    """
    model.to(device=device)

    images, labels = next(iter(dataloader))
    images, labels = images.to(device=device), labels.to(device=device)

    plt.figure(figsize=(10, 8))

    with torch.inference_mode():
        preds = model(images)
        preds = preds.argmax(dim=1)

    for index, image in enumerate(images):
        real = labels[index]
        pred = preds[index]
        text_color = "green" if real == pred else "red"

        plt.subplot(4, 8, index + 1)
        plt.imshow(image.cpu().permute((1, 2, 0)))
        plt.title(
            f"Real: {real_labels[real]}\nPred: {real_labels[pred]}",
            c=text_color,
            fontsize=6,
        )
        plt.axis(False)
