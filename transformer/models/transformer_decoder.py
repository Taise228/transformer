import torch.nn as nn

from transformer.blocks.embedding import TransformerEmbedding
from transformer.layers.decoder_layer import DecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, tokenizer, d_model, num_heads, d_ff, N=6, dropout=0.1, device='gpu', **kwargs):
        """ Transformer Decoder

        Args:
            tokenizer (transformers.AutoTokenizer): tokenizer
            d_model (int): dimension of model
            num_heads (int): number of heads
            d_ff (int): dimension of feed forward
            N (int): number of decoder layers
                default: 6
            dropout (float): dropout rate
                default: 0.1
            device (str): device to use
                default: 'gpu'
            kwargs: keyword arguments listed below;
                max_len (int): maximum length of token sequence. default: 512
                padding_idx (int): index of padding token. default: None
                eps (float): epsilon value for layer normalization. default: 1e-6
        """

        super().__init__()

        # kwargs
        if 'max_len' in kwargs:
            max_len = kwargs['max_len']
        else:
            max_len = 512
        if 'padding_idx' in kwargs:
            padding_idx = kwargs['padding_idx']
        else:
            padding_idx = None
        if 'eps' in kwargs:
            eps = kwargs['eps']
        else:
            eps = 1e-6

        self.embedding = TransformerEmbedding(tokenizer.vocab_size, d_model, max_len=max_len,
                                              padding_idx=padding_idx, dropout=dropout, device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, eps=eps) for _ in range(N)])

    def forward(self, x, enc_output, self_mask, cross_mask):
        """ Forward method of transformer decoder

        Args:
            x (torch.Tensor): input tensor of tokenized input sequence
                shape: (batch_size, dec_input_len)
            enc_output (torch.Tensor): encoder output tensor
                shape: (batch_size, enc_input_len, d_model)
            self_mask (torch.Tensor): self attention mask
                shape: (batch_size, dec_input_len, dec_input_len)
            cross_mask (torch.Tensor): cross attention mask
                shape: (batch_size, dec_input_len, enc_input_len)

        Returns:
            outputs (dict): dictionary of output tensors
                {
                    'output': torch.Tensor, shape: (batch_size, dec_input_len, d_model)
                    'self_attention': list of torch.Tensor, shape: (batch_size, num_heads, dec_input_len, dec_input_len)
                    'cross_attention': list of torch.Tensor, shape: (batch_size, num_heads, dec_input_len, enc_input_len)
                }
        """

        # embedding
        x = self.embedding(x)

        self_attns = []
        cross_attns = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, self_mask, cross_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)

        outputs = {
            'output': x,
            'self_attention': self_attns,
            'cross_attention': cross_attns
        }
        return outputs
