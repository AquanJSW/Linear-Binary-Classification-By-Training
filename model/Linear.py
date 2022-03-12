import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._linear = nn.Linear(2, 1)
    
    def forward(self, x):
        """
        
        Args
        ---
        `x`: Shape (*, 2)

        Return
        ---
        Shape (*, 1)
        """
        y = self._linear(x)
        out = torch.sigmoid(y)

        return out
