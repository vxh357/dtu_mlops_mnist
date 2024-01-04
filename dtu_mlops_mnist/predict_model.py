import torch
import click

from dtu_mlops_mnist.models import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.command()
@click.option('-model_path', prompt='Model path', help='Path to the model.')
@click.option('-data_path', prompt='Data path', help='Path to the data to be processed.')
def predict(model_path, data_path):
    """Predict model on data."""
    print("Predicting model on data")

    net = model.MyNeuralNet(784,10).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    data = torch.load(data_path)

    # Normalize the input data
    data = (data - data.mean()) / data.std()

    pred = net(data.to(device)).cpu().detach()

    click.echo(torch.argmax(pred, dim=1))
    return pred

# def predict(
#     model: torch.nn.Module,
#     data: torch.tensor
# ) -> None:
#     """Run prediction for a given model and dataloader.
    
#     Args:
#         model: model to use for prediction
#         dataloader: dataloader with batches
    
#     Returns
#         Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

#     """

#     return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == """__main__""":
    predict()
