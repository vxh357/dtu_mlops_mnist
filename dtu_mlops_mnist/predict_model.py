import torch
import click
import hydra
from dtu_mlops_mnist.models import model

# Choose the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#@click.command()
#@click.option('-model_path', prompt='Model path', help='Path to the model.')
#@click.option('-data_path', prompt='Data path', help='Path to the data to be processed.')

@hydra.main(config_path="config", config_name="default_config.yaml", version_base=None)
def predict(config):
    """
    Loads a trained neural network model and predicts outputs on given data.

    The function loads a model from the specified path, normalizes the input data,
    and then performs predictions. The predictions are output to the console.

    Args:
        model_path (str): The file path where the trained model is stored.
        data_path (str): The file path where the input data for prediction is stored.

    Returns:
        torch.Tensor: The predictions made by the model on the input data.
    """
    print("Predicting model on data")
    model_hparams = config.model
    train_hparams = config.train

    # Load the neural network model
    net = model.MyNeuralNet(model_hparams, train_hparams["x_dim"], train_hparams["class_num"]).to(device)
    net.load_state_dict(torch.load(model_hparams["model_pred_path"], map_location=torch.device("cpu")))
    net.eval()

    # Load and normalize the input data
    data = torch.load(train_hparams["test_dataset_path"], map_location=torch.device("cpu"))
    data = (data - data.mean()) / data.std()  # Normalize the data

    # Make predictions
    pred = net(data[:][0].to(device)).cpu().detach()  # Move data to device, predict, and detach from graph

    # Output the predictions
    click.echo(torch.argmax(pred, dim=1))
    return pred

# Commented-out code for an alternate predict function
# def predict(
#     model: torch.nn.Module,
#     data: torch.tensor
# ) -> None:
#     """
#     [Commented Out] Alternate version of a prediction function.

#     This function is designed to take a PyTorch model and a tensor of data, and return the predictions.
#     It is currently not in use and serves as an example or template for a different prediction approach.
    
#     Args:
#         model: Model to use for prediction.
#         data: Data tensor for prediction.
    
#     Returns:
#         torch.Tensor: Concatenated predictions for the input data.
#     """

#     return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == "__main__":
    predict()
