"""
Contains the code to help instantiate the a model using CNN
"""
import torch
import torch.nn as nn


class ModelV3(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: int):
        """
        This creates a model with `in_features` nodes
        and predicts `out_features` nodes. The `hidden_layers`
        connect the two nodes.

        :param in_features: The number of features in the image.
            This will likely be color channels. Do NOT multiply by
            height and width as this uses Convolutional Neural
            Networks.
        :param out_features: The number of predictors.
        :param hidden_layers: This connects the input layer and
            the output layer together so feel free to experiment
            with this hyperparameter.
        """
        super().__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=hidden_layers,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            ),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_layers,
                out_channels=hidden_layers,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=0,
            )
        )

        self.layer_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=43264,
                out_features=out_features,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)

        return x3
