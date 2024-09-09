import torch
import torch.nn as nn

from transformer.models.transformer_encoder import TransformerEncoder
from transformer.models.transformer_decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, src_tokenizer, tgt_tokenizer, d_model, num_heads, d_ff, N=6, dropout=0.1, device='gpu', **kwargs):
        """ Transformer model

        Args:
            src_tokenizer (transformers.AutoTokenizer): tokenizer for source language
            tgt_tokenizer (transformers.AutoTokenizer): tokenizer for target language
            d_model (int): dimension of model
            num_heads (int): number of heads
            d_ff (int): dimension of feed forward
            N (int): number of encoder and decoder layers
                default: 6
            dropout (float): dropout rate
                default: 0.1
            device (str): device to use
                default: 'gpu'
            kwargs: keyword arguments listed below;
                max_len (int): maximum length of token sequence. default: 512
                eps (float): epsilon value for layer normalization. default: 1e-6
        """

        super().__init__()

        # kwargs
        if 'max_len' in kwargs:
            max_len = kwargs['max_len']
        else:
            max_len = 512
        if 'eps' in kwargs:
            eps = kwargs['eps']
        else:
            eps = 1e-6

        src_padding_idx = src_tokenizer.pad_token_id
        tgt_padding_idx = tgt_tokenizer.pad_token_id

        self.encoder = TransformerEncoder(src_tokenizer, d_model, num_heads, d_ff, N=N, dropout=dropout, device=device,
                                          max_len=max_len, padding_idx=src_padding_idx, eps=eps)
        self.decoder = TransformerDecoder(tgt_tokenizer, d_model, num_heads, d_ff, N=N, dropout=dropout, device=device,
                                          max_len=max_len, padding_idx=tgt_padding_idx, eps=eps)
        self.head = nn.Linear(d_model, tgt_tokenizer.vocab_size)

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def forward(self, src, tgt):
        """ Forward method of transformer model

        Args:
            src (torch.Tensor): input tensor of tokenized source sequence
                shape: (batch_size, src_len)
            tgt (torch.Tensor): input tensor of tokenized target sequence
                shape: (batch_size, tgt_len)

        Returns:
            outputs (dict): dictionary of output tensors
                {
                    'encoder_output': torch.Tensor, shape: (batch_size, src_len, d_model)
                    'decoder_output': torch.Tensor, shape: (batch_size, tgt_len, d_model)
                    'logits': torch.Tensor, shape: (batch_size, tgt_len, vocab_size)
                    'encoder_attention': list of torch.Tensor, shape: (batch_size, num_heads, tgt_len, src_len)
                    'decoder_self_attention': list of torch.Tensor, shape: (batch_size, num_heads, tgt_len, tgt_len)
                    'decoder_cross_attention': list of torch.Tensor, shape: (batch_size, num_heads, tgt_len, src_len)
                }
        """

        enc_mask = self._generate_mask_pad(src)
        dec_mask = self._generate_square_subsequent_mask(tgt)
        cross_mask = self._generate_mask_pad(src)

        enc_output = self.encoder(src, enc_mask)
        dec_output = self.decoder(tgt, enc_output['output'], dec_mask, cross_mask)
        logits = self.head(dec_output['output'])

        output = {
            'encoder_output': enc_output['output'],
            'decoder_output': dec_output['output'],
            'logits': logits,
            'encoder_attention': enc_output['attention'],
            'decoder_self_attention': dec_output['self_attention'],
            'decoder_cross_attention': dec_output['cross_attention']
        }
        return output
    
    def _generate_square_subsequent_mask(self, x):
        """ Generate square subsequent mask such as below.
        [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]]

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len)

        Returns:
            mask (torch.Tensor): tensor with subsequent mask
                shape: (batch_size, 1, seq_len, seq_len)
        """

        batch_size, seq_len = x.size()
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(1, seq_len, seq_len)
        mask = mask.repeat(batch_size, 1, 1, 1)
        return mask
    
    def _generate_mask_pad(self, x):
        """ Generate mask tensor for padding tokens

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len)

        Returns:
            mask (torch.Tensor): tensor with mask for padding tokens
                shape: (batch_size, 1, 1, seq_len)
        """

        mask = (x != self.src_tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask
