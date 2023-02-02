"""
Contains types for results of certain functions
"""
import torch
from typing import TypedDict, List, Dict


class ModelResult(TypedDict):
    model: str
    loss: float
    acc: float


class TrainTestResult(TypedDict):
    model_states: List[Dict[str, torch.Tensor]]
    optimizer_states: List[Dict[str, torch.Tensor]]
    train: List[ModelResult]
    test: List[ModelResult]
