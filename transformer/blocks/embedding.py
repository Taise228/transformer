import torch
import torch.nn as nn


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, emb_dim, padding_idx=None):
        """ Embedding layer for transformer model

        Args:
            vocab_size (int): size of vocabulary
            emb_dim (int): dimension of embedding
            padding_idx (int): index of padding token
                default: None
        """
        super().__init__(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_dim = emb_dim


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        """ Positional embedding for transformer model
        
        Args
            emb_dim (int): dimension of embedding
            max_len (int): maximum length of token sequence (i.e. the number of tokens in a single input)
                default: 512
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.pe = self._calc_positional_encoding()

    def _calc_positional_encoding(self):
        """ Calculate positional encoding matrix with following formula:
        pe(pos, 2i) = sin(pos / 10000^(2i / emb_dim))
        pe(pos, 2i+1) = cos(pos / 10000^(2i / emb_dim))
        where pos is the position and i is the dimension of embedding

        Returns:
            pe (torch.Tensor): positional encoding matrix. pe[i][j] corresponds to the positional encoding of i-th token's j-th dimension
                shape: (max_len, emb_dim)
        """
        pe = torch.zeros(self.max_len, self.emb_dim)
        pe.requires_grad = False   # positional encoding is not trainable
        position = torch.arange(0, self.max_len).unsqueeze(1).float()   # shape: (max_len, 1)

        even = torch.arange(0, self.emb_dim, 2).float()   # shape: (emb_dim / 2)
        even = torch.pow(10000, even / self.emb_dim)   # shape: (emb_dim / 2)
        pe[:, 0::2] = torch.sin(position / even)  # shape: (max_len, emb_dim / 2)
        odd = torch.arange(1, self.emb_dim, 2).float()
        odd = torch.pow(10000, odd / self.emb_dim)
        pe[:, 1::2] = torch.cos(position / odd)

        return pe
    
    def forward(self, x):
        """ Add positional encoding to input tensor

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len, emb_dim)

        Returns:
            x (torch.Tensor): input tensor with positional encoding
                shape: (batch_size, seq_len, emb_dim)
        """
        x = x + self.pe[:x.size(1)]
        return x
    

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len=512, padding_idx=None, dropout=0.1):
        """ Embedding layer for transformer model

        Args:
            vocab_size (int): size of vocabulary
            emb_dim (int): dimension of embedding
            max_len (int): maximum length of token sequence (i.e. the number of tokens in a single input)
                default: 512
            padding_idx (int): index of padding token
                default: None
            dropout (float): dropout rate. If dropout is 0, dropout is not applied.
                default: 0.1
        """
        super().__init__()
        self.emb = Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.pos_emb = PositionalEmbedding(emb_dim, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim

    def forward(self, x):
        """ Forward method of transformer embedding layer

        Args:
            x (torch.Tensor): input tensor
                shape: (batch_size, seq_len)

        Returns:
            x (torch.Tensor): input tensor with positional encoding
                shape: (batch_size, seq_len, emb_dim)
        """
        x = self.emb(x)
        x = x * (self.emb_dim ** 0.5)   # scale embedding
        x = self.pos_emb(x)
        x = self.dropout(x)
        return x
    

# Usage
if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text = "Hello, world! This is transformer embedding layer."
    tokens = tokenizer(text, return_tensors='pt')['input_ids']
    print(tokens)   # shape: (1, token_length)

    embedder = TransformerEmbedding(tokenizer.vocab_size, 768)
    embedded = embedder(tokens)
    print(embedded.shape)   # torch.Size([1, token_length, 768])
