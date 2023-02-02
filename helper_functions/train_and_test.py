"""
Contains functionality for common tasks for each model
"""
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from helper_functions.result_types import ModelResult, TrainTestResult
from typing import Callable


def train_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: Callable[..., float],
        device: torch.device,
) -> ModelResult:
    """
    This trains a model for one step. Use in a for loop for
    convenience purposes. For a complete train and test loop,
    use `train_and_test` instead.

    :param model: A model to be trained
    :param dataloader: A dataloader that contains the data
        for training the model
    :param loss_fn: A loss function that calculates the loss
        of a training model
    :param optimizer: An optimizer that optimizes the
        parameters of a model to better itself
    :param accuracy_fn: A function that calculates the
        accuracy of a model in predicting the result.
    :param device: The device in which the model should
        be trained. Ideally, this is a constant variable
    :return: A `ModelResult` for training data
    """
    model.train()

    total_loss = 0
    total_acc = 0

    for batch, (image, label) in enumerate(dataloader):
        image, label = image.to(device=device), label.to(device=device)

        preds_logits = model(image)
        preds = preds_logits.argmax(dim=1)

        loss = loss_fn(preds_logits, label)
        total_loss += loss
        acc = accuracy_fn(label.cpu(), preds.cpu())
        total_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 30 == 0:
            print(f"Batch {batch}: ({len(image) * batch} / {len(dataloader.dataset)})")

    return {
        "model": model.__class__.__name__,
        "loss": total_loss / len(dataloader),
        "acc": total_acc / len(dataloader),
    }


def test_step(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,  # For consistency purposes
        accuracy_fn: Callable[..., float],
        device: torch.device,
) -> ModelResult:
    """
    This tests or validates a model for one step. Use in a
    for loop for convenience purposes. For a complete train
    and test loop, use `train_and_test` instead.

    :param model: A model to be tested or validated
    :param dataloader: A dataloader that contains the data
        for testing or validating the model
    :param loss_fn: A loss function that calculates the loss
        of a training model
    :param optimizer: An optimizer that optimizes the
        parameters of a model to better itself. Note that
        this will not be used by the function and *is in
        here for consistency purposes only*.
    :param accuracy_fn: A function that calculates the
        accuracy of a model in predicting the result.
    :param device: The device in which the model should
        be trained. Ideally, this is a constant variable
    :return: A `ModelResult` for testing or validation data
    """
    model.eval()

    total_loss = 0
    total_acc = 0

    for image, label in dataloader:
        with torch.inference_mode():
            image, label = image.to(device=device), label.to(device=device)

            preds_logits = model(image)
            preds = preds_logits.argmax(dim=1)

            loss = loss_fn(preds_logits, label)
            total_loss += loss
            acc = accuracy_fn(label.cpu(), preds.cpu())
            total_acc += acc

    return {
        "model": model.__class__.__name__,
        "loss": total_loss / len(dataloader),
        "acc": total_acc / len(dataloader),
    }


def train_and_test(
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        accuracy_fn: Callable[..., float],
        device: torch.device,
        epochs: int,
        previous: TrainTestResult,
) -> TrainTestResult:
    """
    This function combines both train and test functions in
    a single function. Use this so you don't have to write
    for loops over and over again.

    Note that this function does not contain a mechanism for
    early stopping. To find the most optimal model, you must
    manually find it.

    :param model: A model to be trained and tested
    :param dataloader: A dataloader that contains the data
        for testing or validating the model
    :param train_dataloader: A dataloader that contains the data
        for training the model
    :param test_dataloader: A dataloader that contains the data
        for testing or validating the model
    :param loss_fn: A loss function that calculates the loss
        of a training model
    :param optimizer: An optimizer that optimizes the
        parameters of a model to better itself. Note that
        this will not be used by the function and *is in
        here for consistency purposes only*.
    :param accuracy_fn: A function that calculates the
        accuracy of a model in predicting the result.
    :param device: The device in which the model should
        be trained. Ideally, this is a constant variable
    :param epochs: Number of epochs the model should be run
        for
    :param previous: A `TrainTestResult` of the previous run.
        This is useful if you want to resume training at a said
        point rather than starting over again.
    :return: A `TrainTestResult` for the said number of epochs.
    """

    model_states, optimizer_states = [], []
    train_results, test_results = [], []
    start_epoch = len(previous['test'])

    model.to(device=device)

    for epoch in tqdm(range(start_epoch, epochs)):
        train_details = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        train_loss = train_details['loss']
        train_acc = train_details['acc']

        test_details = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        test_loss = test_details['loss']
        test_acc = test_details['acc']

        train_results.append(train_details)
        test_results.append(test_details)
        model_states.append(model.state_dict())
        optimizer_states.append(optimizer.state_dict())

        print(f"Epoch {epoch}")
        print(f"--- Train | Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"--- Test  | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    return {
        "model_states": [*previous['model_states'], *model_states],
        "optimizer_states": [*previous['optimizer_states'], *optimizer_states],
        "train": [*previous['train'], *train_results],
        "test": [*previous['test'], *test_results],
    }
