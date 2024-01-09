import os
import pytest
import torch
from hydra import compose, initialize

from dtu_mlops_mnist.models import model
from dtu_mlops_mnist.predict_model import predict
from tests import _PATH_DATA


# Test model output dimensions on random input data
@pytest.mark.parametrize("test_data", [torch.randn(1, 1, 28, 28), torch.randn(20, 1, 28, 28)])
def test_model(test_data):
    """
    Test the output dimensions of the neural network model on random input data.

    This function creates a neural network model with specified hyperparameters and checks the dimensions
    of the model's output on random input data.

    Args:
        test_data (torch.Tensor): Random input data for testing.

    Raises:
        AssertionError: If the model's output shape does not match the expected shape, it raises an assertion error.
    """

    # Define model hyperparameters
    model_hparams = {"l1_dim": 256, "l2_dim": 128, "l3_dim": 64, "dropout_rate": 0.2}

    # Initialize the neural network model
    net = model.MyNeuralNet(model_hparams, 784, 10)

    # Get the model's output on the test data
    y = net(test_data)

    # Check if the model's output shape matches the expected shape
    assert y.shape == torch.Size([test_data.shape[0], 10]), "Model output shape is incorrect"


# Uncomment the following test if needed
# def test_predict_model():
#     """Test predict_model function."""
#     with initialize(version_base=None, config_path=f"../mlOps_mnist/config", job_name="test_training"):
#         config = compose(config_name="default_config.yaml")
#     pred = predict(config)
#     assert pred.shape == torch.Size([5000, 10]), "Model output shape is incorrect"


# Test that the model raises an error on input with the wrong shape
def test_error_on_wrong_shape():
    """
    Test that the model raises an error when provided with input of the wrong shape.

    This function initializes the neural network model and checks if it raises a ValueError with the expected message
    when provided with input data that does not have the correct shape.

    Raises:
        AssertionError: If the model does not raise the expected error, it raises an assertion error.
    """

    # Define model hyperparameters
    model_hparams = {"l1_dim": 256, "l2_dim": 128, "l3_dim": 64, "dropout_rate": 0.2}

    # Initialize the neural network model
    net = model.MyNeuralNet(model_hparams, 784, 10)

    # Try to pass input data with an incorrect shape and check for the expected error
    with pytest.raises(ValueError, match="Expected input to be a 4D tensor"):
        net(torch.randn(1, 2, 3))
