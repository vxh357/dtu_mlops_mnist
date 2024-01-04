import torch
from torch import nn
import torch.nn.functional as F

class MyNeuralNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.dr1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dr2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dr3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(64, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = x.view(x.shape[0], -1)
        
        x = self.dr1(F.relu(self.fc1(x)))
        x = self.dr2(F.relu(self.fc2(x)))
        x = self.dr3(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
