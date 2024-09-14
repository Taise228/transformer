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
    
    def inference_batch(self, src, max_len=128):
        """ Inference method of transformer model.
        Generate target sequence given source sequence.

        Args:
            src (torch.Tensor): input tensor of tokenized source sequence
                shape: (batch_size, src_len)
            max_len (int): maximum length of output sequence
                default: 128

        Returns:
            outputs (dict): dictionary of output tensors
                {
                    'predictions': torch.Tensor, shape: (batch_size, tgt_len)
                    'encoder_attention': list of torch.Tensor, shape: (batch_size, num_heads, src_len, src_len)
                    'decoder_self_attention': list of torch.Tensor, shape: (batch_size, num_heads, tgt_len, tgt_len)
                    'decoder_cross_attention': list of torch.Tensor, shape: (batch_size, num_heads, tgt_len, src_len)
                }
        """
        assert len(src.size()) == 2, f'Input shape should be (batch_size, src_len), but got {src.size()}'

        batch_size = src.size(0)
        tgt = torch.ones((batch_size, 1), dtype=torch.long, device=src.device) * self.tgt_tokenizer.cls_token_id
        # first token is CLS token

        for _ in range(max_len):
            dec_mask = self._generate_square_subsequent_mask(tgt)
            cross_mask = self._generate_mask_pad(src)

            enc_output = self.encoder(src, cross_mask)
            dec_output = self.decoder(tgt, enc_output['output'], dec_mask, cross_mask)
            logits = self.head(dec_output['output'])

            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)   # shape: (batch_size, 1)
            tgt = torch.cat([tgt, next_token], dim=-1)
            if torch.eq(next_token, self.tgt_tokenizer.sep_token_id).all():
                break

        outputs = {
            'predictions': tgt,
            'encoder_attention': enc_output['attention'],
            'decoder_self_attention': dec_output['self_attention'],
            'decoder_cross_attention': dec_output['cross_attention']
        }
        return outputs
    
    def inference(self, src, max_len=128):
        """ Inference method of transformer model.
        Generate target sequence given source sequence.

        Args:
            src (torch.Tensor): input tensor of tokenized source sequence
                shape: (src_len,)
            max_len (int): maximum length of output sequence
                default: 128

        Returns:
            outputs (dict): dictionary of output tensors
                {
                    'predictions': torch.Tensor, shape: (tgt_len,)
                    'encoder_attention': list of torch.Tensor, shape: (num_heads, src_len, src_len)
                    'decoder_self_attention': list of torch.Tensor, shape: (num_heads, tgt_len, tgt_len)
                    'decoder_cross_attention': list of torch.Tensor, shape: (num_heads, tgt_len, src_len)
                }
        """

        assert len(src.size()) == 1, f'Input shape should be (src_len,), but got {src.size()}'
        src = src.unsqueeze(0)
        outputs = self.inference_batch(src, max_len)
        outputs['predictions'] = outputs['predictions'].squeeze(0)
        outputs['encoder_attention'] = [attn.squeeze(0) for attn in outputs['encoder_attention']]
        outputs['decoder_self_attention'] = [attn.squeeze(0) for attn in outputs['decoder_self_attention']]
        outputs['decoder_cross_attention'] = [attn.squeeze(0) for attn in outputs['decoder_cross_attention']]

        return outputs
