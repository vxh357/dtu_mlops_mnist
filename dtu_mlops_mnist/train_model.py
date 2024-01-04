import click
import torch
from torch import nn
from dtu_mlops_mnist.models import model
import os
import datetime
import matplotlib.pyplot as plt

# Choose the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option("--lr", default=1e-3, help="Learning rate to use for training")
@click.option("--batch_size", default=256, help="Batch size to use for training")
@click.option("--num_epochs", default=20, help="Number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """
    Trains a neural network model on the MNIST dataset.

    This function sets up a training loop for the MNIST dataset using a specified learning rate, batch size,
    and number of epochs. It tracks the loss across epochs and saves the trained model and a loss plot.

    Args:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs for which the model will be trained.

    The function saves the trained model and a loss plot in designated directories.
    """
    print("Training day and night")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")

    # Initialize the neural network model
    net = model.MyNeuralNet(784, 10).to(device)

    # Load the training dataset
    train_set = torch.load("data/processed/train_dataset.pt")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # List to store loss values for each epoch
    loss_list = []

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()      # Zero the gradients before calculation
            x, y = batch               # Get input and target batches
            x = x.to(device)           # Move input to computation device
            y = y.to(device)           # Move target to computation device
            y_pred = net(x)            # Forward pass through the model
            loss = loss_fn(y_pred, y)  # Calculate loss
            loss.backward()            # Backward pass to calculate gradients
            optimizer.step()           # Update model parameters

        print(f"Epoch {epoch} Loss {loss}")
        loss_list.append(loss.cpu().detach().numpy())  # Store loss value

    # Save the trained model
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_savepath = "models/" + timestamp
    os.makedirs(model_savepath)
    torch.save(net.state_dict(), f"{model_savepath}/model.pth")

    # Plot and save the loss graph
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss over Epochs (Model from {timestamp})")

    fig_savepath = "reports/figures/" + timestamp
    os.makedirs(fig_savepath)
    plt.savefig(f"{fig_savepath}/loss.pdf")

if __name__ == "__main__":
    train()
