import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """ Position-wise feed forward network
        Activation function: ReLU

        Args:
            d_model (int): dimension of model
            d_ff (int): dimension of feed forward
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """ Forward method of position-wise feed forward network

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len, d_model)

        Returns:
            x (torch.Tensor): output tensor
                shape: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x