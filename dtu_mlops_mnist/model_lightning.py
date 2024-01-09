import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
import wandb

class MyNeuralNet(LightningModule):
    """
    A basic neural network model built using PyTorch Lightning.

    This class defines a simple feedforward neural network with configurable layers and dropout rates. 
    It includes methods for the forward pass, training step, and optimizer configuration.

    Attributes:
        backbone (nn.Sequential): The sequential model forming the main part of the neural network.
        classifier (nn.Sequential): The final classifier layer of the network.
        criteriun (nn.Module): The loss function used for training the model.

    Args:
        model_hparams (dict): A dictionary containing hyperparameters for the model, such as layer 
        dimensions and dropout rate.
        in_features (int): The number of input features for the model.
        out_features (int): The number of output features (or classes) for the model.
    """

    def __init__(self, model_hparams: dict, in_features: int, out_features: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, model_hparams["l1_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
            nn.Linear(model_hparams["l1_dim"], model_hparams["l2_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
            nn.Linear(model_hparams["l2_dim"], model_hparams["l3_dim"]),
            nn.Dropout(p=model_hparams["dropout_rate"]),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_hparams["l3_dim"], out_features),
            nn.LogSoftmax(dim=1),
        )

        self.criteriun = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor expected to be of shape [N, in_features], where N is the batch size.

        Returns:
            torch.Tensor: The output tensor of shape [N, out_features], where N is the batch size.
        """
        x = x.view(x.shape[0], -1)  # Flatten the input tensor
        x = self.backbone(x)  # Pass through the backbone layers
        x = self.classifier(x)  # Pass through the classifier layer
        return x

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Args:
            batch: The output of your DataLoader. A tuple consisting of an input tensor and its corresponding target tensor.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss value for the current training batch.
        """
        data, target = batch  # Unpack the batch
        preds = self(data)  # Perform forward pass
        loss = self.criteriun(preds, target)  # Calculate loss
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        # self.logger.experiment is the same as wandb.log
        self.logger.experiment.log({'logits': wandb.Histrogram(preds)})
        
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer used for training.

        Returns:
            The optimizer to be used for training. Here, Adam optimizer is used.
        """
        return optim.Adam(self.parameters(), lr=1e-3)