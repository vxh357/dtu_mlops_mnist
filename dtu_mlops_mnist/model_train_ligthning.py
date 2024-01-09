import datetime
import logging
import os
import hydra
import model_lightning  # Ensure this module is properly imported
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger(__name__)
log.info(f"Cuda available: {torch.cuda.is_available()}")


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):
    """
    Trains a Feedforward Neural Network (FNN) on the MNIST dataset.

    This function initializes and trains a neural network model defined in 'model_lightning'.
    The training parameters, model configuration, and other settings are controlled by a
    configuration file managed by Hydra.

    The training process includes early stopping for convergence and logs the training progress
    to Weights & Biases (wandb).

    Args:
        config: Configuration object containing training and model parameters.

    The function saves the trained model and optionally, a loss plot. It doesn't return any value.
    """
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")
    model_hparams = config.model
    train_hparams = config.train
    torch.manual_seed(train_hparams["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the neural network model
    net = model_lightning.MyNeuralNet(model_hparams, train_hparams["x_dim"], train_hparams["class_num"]).to(device)

    # Load the training dataset and create a DataLoader
    train_set = torch.load(train_hparams["dataset_path"])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_hparams["batch_size"])

    # Set up early stopping based on the training loss
    early_stopping_callback = EarlyStopping(monitor="loss", patience=3, verbose=True, mode="min")

    # Configure the PyTorch Lightning trainer
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=train_hparams["n_epochs"],
        limit_train_batches=0.2,
        callbacks=[early_stopping_callback],
        accelerator="auto",
        logger=WandbLogger(project="fashion_mnist_dtu_mlops"),  # logger=pl.loggers.WandbLogger(project="dtu_mlops")
        precision="16-mixed",
        profiler="advanced",
    )

    # Start the training process
    trainer.fit(net, train_dataloader)

    # Save the trained model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_savepath = os.path.join("models", timestamp)
    os.makedirs(model_savepath, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(model_savepath, "model.pth"))
    log.info(f"Model saved to {os.path.join(model_savepath, 'model.pth')}")

    # Optional: Code to plot and save the loss graph can be added here


if __name__ == "__main__":
    train()
