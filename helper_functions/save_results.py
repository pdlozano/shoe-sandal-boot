"""
Contains helper functions for saving and loading
"""
import torch
from pathlib import Path
from helper_functions.result_types import TrainTestResult


def save_checkpoint(results: TrainTestResult, path: Path) -> None:
    """
    Saves a checkpoint of the model.

    :param results: A `TrainTestResult` of the model you want
        saved.
    :param path: A `Path` of where the `.pt` file should be
        saved.
    :return: Returns nothing
    """
    torch.save(obj=results, f=path)
