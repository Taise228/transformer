import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """ Layer normalization

        Args:
            d_model (int): dimension of model
            eps (float): epsilon value
                default: 1e-6
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # learnable parameter
        self.beta = nn.Parameter(torch.zeros(d_model))   # learnable parameter
        self.eps = eps

    def forward(self, x):
        """ Forward method of layer normalization.
        Values of each feature are normalized to have mean of 0 and standard deviation of 1.
        Here, 'feature' corresponds to each token. So calc mean and std around d_model axis.

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len, d_model)
        Returns:
            x (torch.Tensor): output tensor
                shape: (batch_size, seq_len, d_model)
        """

        mean = x.mean(-1, keepdim=True)   # mean of each feature. shape: (batch_size, seq_len, 1)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
