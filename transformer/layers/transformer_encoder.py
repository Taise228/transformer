import torch.nn as nn

from transformer.blocks.embedding import TransformerEmbedding
from transformer.layers.encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, tokenizer, d_model, num_heads, d_ff, N=6, dropout=0.1, **kwargs):
        """ Transformer Encoder

        Args:
            tokenizer (transformers.AutoTokenizer): tokenizer
            d_model (int): dimension of model
            num_heads (int): number of heads
            d_ff (int): dimension of feed forward
            N (int): number of encoder layers
                default: 6
            dropout (float): dropout rate
                default: 0.1
            kwargs: keyword arguments listed below;
                max_len (int): maximum length of token sequence. default: 512
                padding_idx (int): index of padding token. default: None
                eps (float): epsilon value for layer normalization. default: 1e-6
        """
        super().__init__()

        # kwargs
        if 'max_len' in kwargs:   # maximum length of token sequence
            max_len = kwargs['max_len']
        else:
            max_len = 512
        if 'padding_idx' in kwargs:   # index of padding token
            padding_idx = kwargs['padding_idx']
        else:
            padding_idx = None
        if 'eps' in kwargs:   # epsilon value for layer normalization
            eps = kwargs['eps']
        else:
            eps = 1e-6

        self.embedding = TransformerEmbedding(tokenizer.vocab_size, d_model, max_len=max_len, padding_idx=padding_idx, dropout=dropout)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, eps=eps) for _ in range(N)])

    def forward(self, x, mask):
        """ Forward method of transformer encoder

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len)
            mask (torch.Tensor): attention mask
                shape: (batch_size, seq_len, seq_len)

        Returns:
            outputs (dict): dictionary of output tensors
                {
                    'output': torch.Tensor, shape: (batch_size, seq_len, d_model)
                    'attention': list of torch.Tensor, shape: (batch_size, num_heads, seq_len, seq_len)
                }
        """

        x = self.embedding(x)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)

        outputs = {
            'output': x,
            'attention': attns
        }
        return outputs
