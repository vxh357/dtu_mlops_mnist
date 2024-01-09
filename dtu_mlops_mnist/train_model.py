import datetime
import logging
import os

import click
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from torch import nn
from dtu_mlops_mnist.models import model
import matplotlib.pyplot as plt

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="fashion_mnist_dtu_mlops",
    entity="vxh357-dtu_mlops",)

log = logging.getLogger(__name__)

# Choose the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info(f"Cuda available: {device}")

#@click.command()
#@click.option("--lr", default=1e-3, help="Learning rate to use for training")
#@click.option("--batch_size", default=256, help="Batch size to use for training")
#@click.option("--num_epochs", default=20, help="Number of epochs to train for")

@hydra.main(version_base="1.1", config_path="config", config_name="default_config.yaml")
def train(config: DictConfig) -> None:
    """
    Trains a neural network model on the MNIST dataset.

    This function sets up a training loop for the MNIST dataset using parameters from the given configuration.
    It tracks the loss across epochs and saves the trained model and a loss plot in designated directories.

    The function is designed to be used with Hydra for configuration management, allowing parameters such as learning
    rate, batch size, and number of epochs to be specified via configuration files or command line overrides.

    Args:
        config (OmegaConf): Configuration object containing parameters for training.

    The function does not return any value.
    """

    log.info("Training day and night")
    log.info(f"Configuration: \n {OmegaConf.to_yaml(config)}")
    model_hparams = config.model
    train_hparams = config.train
    torch.manual_seed(train_hparams["seed"])

    # Initialize the neural network model
    net = model.MyNeuralNet(model_hparams, train_hparams["x_dim"], train_hparams["class_num"]).to(device)

    # Load the training dataset
    train_set = torch.load(train_hparams["dataset_path"])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_hparams["batch_size"])

    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=train_hparams["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # List to store loss values for each epoch
    loss_list = []

    # Training loop
    for epoch in range(train_hparams["n_epochs"]):
        for batch in train_dataloader:
            optimizer.zero_grad()      # Zero the gradients before calculation
            x, y = batch               # Get input and target batches
            x = x.to(device)           # Move input to computation device
            y = y.to(device)           # Move target to computation device
            y_pred = net(x)            # Forward pass through the model
            loss = loss_fn(y_pred, y)  # Calculate loss
            loss.backward()            # Backward pass to calculate gradients
            optimizer.step()           # Update model parameters

        log.info(f"Epoch {epoch} Loss {loss}")
        loss_list.append(loss.cpu().detach().numpy())  # Store loss value
        wandb.log({"loss": loss})

    # Save the trained model
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_savepath = "models/" + timestamp
    os.makedirs(model_savepath)
    torch.save(net.state_dict(), f"{model_savepath}/model.pth")
    log.info(f"Model saved to {model_savepath}/model.pth")

    # Plot and save the loss graph
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs (Model from {timestamp})")
    
    fig_savepath = "reports/figures/" + timestamp
    os.makedirs(fig_savepath)
    plt.savefig(f"{fig_savepath}/loss.pdf")
    log.info(f"Loss figure saved to {fig_savepath}/loss.pdf")
    
    return net, loss_list, model_savepath, fig_savepath

    # wandb.log({"Test loss plot": fig})

if __name__ == "__main__":
    train()
