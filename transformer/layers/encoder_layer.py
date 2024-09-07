import torch.nn as nn

from transformer.blocks.attention import MultiHeadAttention
from transformer.blocks.layer_norm import LayerNormalization
from transformer.blocks.pw_ffn import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, eps=1e-6):
        """Single encoder layer of transformer model
        
        Args:
            d_model (int): inner dimension of model
            num_heads (int): number of heads in multi-head attention
            d_ff (int): hidden dimension of feed forward
            dropout (float): dropout rate
                default: 0.1
            eps (float): epsilon value for layer normalization
                default: 1e-6
        """

        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model, eps=eps)
        self.dropout1 = nn.Dropout(p=dropout)

        self.pw_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNormalization(d_model, eps=eps)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """Forward method of encoder layer
        
        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len, d_model)
            mask (torch.Tensor): attention mask
                shape: (batch_size, seq_len, seq_len)
        
        Returns:
            x (torch.Tensor): output tensor
                shape: (batch_size, seq_len, d_model)
            attn (torch.Tensor): attention weights
                shape: (batch_size, num_heads, seq_len, seq_len)
        """

        # self attention
        context, attn = self.attention(x, x, x, mask)   # Q, K, V are all x
        context = self.dropout1(context)
        x = self.norm1(x + context)   # Residual connection and layer normalization
        # x.shape == context.shape == (batch_size, seq_len, d_model)

        # position-wise feed forward network
        output = self.pw_ffn(x)   # shape: (batch_size, seq_len, d_model)
        output = self.dropout2(output)
        x = self.norm2(x + output)   # Residual connection and layer normalization

        return x, attn
