import glob
import os
import pytest
from hydra import compose, initialize
from dtu_mlops_mnist.train_model import train
from tests import _PATH_DATA

# Define a pytest test function
@pytest.mark.skipif(not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pt"), reason="Data files not found")
def test_train():
    """
    Test the training process.

    This function tests the training process by running the training script with a specified configuration.
    It checks if the trained model and loss plot are saved, and if the loss list is not empty.

    Raises:
        AssertionError: If any of the checks fail, it raises an assertion error with an appropriate message.
    """

    # Initialize Hydra with the test configuration
    with initialize(version_base=None, config_path="../dtu_mlops_mnist/config", job_name="test_training"):
        config = compose(config_name="default_config.yaml")
    
    # Run the training and get the returned values
    _, loss_list, model_savepath, fig_savepath = train(config)

    # Check if the trained model is saved
    assert os.path.exists(f"{model_savepath}/model.pth"), "Model was not saved"

    # Check if the loss plot is saved
    assert os.path.exists(f"{fig_savepath}/loss.pdf"), "Loss plot was not saved"

    # Check if the loss list is not empty
    assert len(loss_list) > 0, "Loss list is empty"

    # Clean up temporary files and directories created during testing
    files = glob.glob(f"{model_savepath}/*")
    for f in files:
        os.remove(f)
    os.removedirs(f"{model_savepath}/")

    files = glob.glob(f"{fig_savepath}/*")
    for f in files:
        os.remove(f)
    os.removedirs(f"{fig_savepath}/")
