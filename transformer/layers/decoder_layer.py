import torch.nn as nn

from transformer.blocks.attention import MultiHeadAttention
from transformer.blocks.layer_norm import LayerNormalization
from transformer.blocks.pw_ffn import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, eps=1e-6):
        """Single decoder layer of transformer model
        
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
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model, eps=eps)
        self.dropout1 = nn.Dropout(p=dropout)

        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model, eps=eps)
        self.dropout2 = nn.Dropout(p=dropout)

        self.pw_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNormalization(d_model, eps=eps)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        """Forward method of decoder layer

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, dec_input_len, d_model)
            enc_output (torch.Tensor): encoder output tensor
                shape: (batch_size, enc_input_len, d_model)
            self_mask (torch.Tensor): self attention mask
                shape: (batch_size, dec_input_len, dec_input_len)
            cross_mask (torch.Tensor): cross attention mask
                Q comes from x. K and V come from enc_output.
                So shape of Q * K.T should be (batch_size, dec_input_len, enc_input_len)
                shape: (batch_size, dec_input_len, enc_input_len)

        Returns:
            x (torch.Tensor): output tensor
                shape: (batch_size, dec_input_len, d_model)
            self_attn (torch.Tensor): self attention weights
                shape: (batch_size, num_heads, dec_input_len, dec_input_len)
            cross_attn (torch.Tensor): cross attention weights
                shape: (batch_size, num_heads, dec_input_len, enc_input_len)
        """

        # self attention
        self_context, self_attn = self.self_attention(x, x, x, self_mask)
        self_context = self.dropout1(self_context)
        x = self.norm1(x + self_context)
        # x.shape: (batch_size, dec_input_len, d_model)

        # cross attention
        cross_context, cross_attn = self.cross_attention(x, enc_output, enc_output, cross_mask)
        cross_context = self.dropout2(cross_context)
        x = self.norm2(x + cross_context)
        # x.shape: (batch_size, dec_input_len, d_model)

        # position-wise feed forward network
        output = self.pw_ffn(x)
        output = self.dropout3(output)
        x = self.norm3(x + output)
        # x.shape: (batch_size, dec_input_len, d_model)

        return x, self_attn, cross_attn
    