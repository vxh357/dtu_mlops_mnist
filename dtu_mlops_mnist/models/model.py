import torch
from torch import nn
import torch.nn.functional as F


class MyNeuralNet(nn.Module):
    """
    A basic feedforward neural network with dropout layers.

    This neural network consists of four fully connected (fc) layers with ReLU activations and dropout layers for regularization.
    The final output is passed through a log softmax layer.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        dr1 (nn.Dropout): Dropout layer after the first fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        dr2 (nn.Dropout): Dropout layer after the second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        dr3 (nn.Dropout): Dropout layer after the third fully connected layer.
        fc4 (nn.Linear): Fourth fully connected layer, producing the final output.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """

    def __init__(self, model_hparams: dict, in_features: int, out_features: int) -> None:
        super().__init__()
        # Define the first fully connected layer and corresponding dropout
        self.fc1 = nn.Linear(
            in_features, model_hparams["l1_dim"]
        )  # First layer with 'in_features' inputs and 256 outputs
        self.dr1 = nn.Dropout(p=model_hparams["dropout_rate"])  # Dropout for regularization with 20% probability

        # Define the second fully connected layer and corresponding dropout
        self.fc2 = nn.Linear(
            model_hparams["l1_dim"], model_hparams["l2_dim"]
        )  # Second layer with 256 inputs and 128 outputs
        self.dr2 = nn.Dropout(p=model_hparams["dropout_rate"])  # Dropout for regularization with 20% probability

        # Define the third fully connected layer and corresponding dropout
        self.fc3 = nn.Linear(
            model_hparams["l2_dim"], model_hparams["l3_dim"]
        )  # Third layer with 128 inputs and 64 outputs
        self.dr3 = nn.Dropout(p=model_hparams["dropout_rate"])  # Dropout for regularization with 20% probability

        # Define the fourth and final fully connected layer
        self.fc4 = nn.Linear(
            model_hparams["l3_dim"], out_features
        )  # Final layer with 64 inputs and 'out_features' outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network.

        The input tensor is passed through each layer sequentially with ReLU activation functions
        applied to the outputs of the first three fully connected layers. Each of these is followed
        by a dropout layer. The final output is passed through a log softmax layer.

        Args:
            x (torch.Tensor): Input tensor expected to be of shape [N, 1, 28, 28], where N is the batch size.

        Returns:
            torch.Tensor: Output tensor of shape [N, out_features], where N is the batch size.
        """

        if x.ndim != 4:
            # Check if the input tensor is not 4-dimensional (batch size, channels, height, width)
            raise ValueError("Expected input to be a 4D tensor")

        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            # Check if each sample in the batch does not have the expected shape of [1 channel, 28 height, 28 width]
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        # Flatten the input tensor to fit the expected input shape of the first fully connected layer
        x = x.view(x.shape[0], -1)

        # Apply layers sequentially: fully connected -> ReLU -> Dropout
        x = self.dr1(F.relu(self.fc1(x)))  # Apply fc1, ReLU activation, then dropout
        x = self.dr2(F.relu(self.fc2(x)))  # Apply fc2, ReLU activation, then dropout
        x = self.dr3(F.relu(self.fc3(x)))  # Apply fc3, ReLU activation, then dropout

        # Apply the final fully connected layer and log softmax
        x = F.log_softmax(self.fc4(x), dim=1)  # Apply fc4 and log softmax for the output

        return x
