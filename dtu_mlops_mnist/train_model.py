import click
import torch
from torch import nn
from mlOps_mnist.models import model
import os
import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=256, help="batch size to use for training")
@click.option("--num_epochs", default=20, help="number of epochs to train for")
def train(lr, batch_size, num_epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    print(batch_size)

    # TODO: Implement training loop here
    net = model.MyNeuralNet(784,10).to(device)
    train_set = torch.load("data/processed/train_dataset.pt")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loss_list = []
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")
        loss_list.append(loss.cpu().detach().numpy())

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_savepath = "models/" + timestamp 
    os.makedirs(model_savepath)

    torch.save(net.state_dict(), f"{model_savepath}/model.pth")

    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss for model from {timestamp}")

    fig_savepath = "reports/figures/" + timestamp
    os.makedirs(fig_savepath)
    plt.savefig(f"{fig_savepath}/loss.pdf")

if __name__ == "__main__":
    train()
