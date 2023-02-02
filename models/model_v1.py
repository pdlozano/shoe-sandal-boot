"""
Contains the code to help instantiate the linear model with `ReLU`
"""
import torch
import torch.nn as nn


class ModelV1(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: int):
        """
        This creates a model with `in_features` nodes
        and predicts `out_features` nodes. The `hidden_layers`
        connect the two nodes.

        The difference between `ModelV0` and `ModelV1` is that
        `ModelV1` has a ReLU layer that allows it to
        theoretically perform better.

        :param in_features: The number of features in the image.
            This will likely be color channels multiplied by
            height and width.
        :param out_features: The number of predictors.
        :param hidden_layers: This connects the input layer and
            the output layer together so feel free to experiment
            with this hyperparameter.
        """
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=in_features,
                out_features=hidden_layers,
            ),
            nn.ReLU(),
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(
                in_features=hidden_layers,
                out_features=out_features,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)

        return x2
