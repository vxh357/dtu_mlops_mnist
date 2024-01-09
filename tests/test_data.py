import os
import hydra
import pytest
import torch

from tests import _PATH_DATA

# Define a pytest test function
@pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pt"), reason="Data files not found")
def test_data():
    """
    Test the loaded data from the processed dataset files.

    This function loads the training and testing datasets from processed files and performs various checks on them.
    It ensures that the datasets have the correct number of samples, dimensions of samples, and classes.

    Raises:
        AssertionError: If any of the checks fail, it raises an assertion error with an appropriate message.
    """

    # Load the training and testing datasets
    train_set = torch.load(f"{_PATH_DATA}/processed/train_dataset.pt")
    test_set = torch.load(f"{_PATH_DATA}/processed/test_dataset.pt")

    # Check if the training dataset has the correct number of samples
    assert len(train_set) == 45000, "Dataset did not have the correct number of samples"

    # Check if the testing dataset has the correct number of samples
    assert len(test_set) == 5000, "Dataset did not have the correct number of samples"

    # Check the dimensions of samples in both datasets
    assert (
        train_set[:][0].shape == torch.Size([45000, 1, 28, 28]) and
        test_set[:][0].shape == torch.Size([5000, 1, 28, 28])
    ), "The dimensions of samples are not correct"

    # Check if the dataset contains all the expected classes (0 to 9)
    assert (
        (torch.unique(train_set[:][1]) == torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).sum() == torch.tensor(10)
    ), "The dataset does not contain all the classes"
