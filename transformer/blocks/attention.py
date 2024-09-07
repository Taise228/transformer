import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention with optional mask
    Take inputs divided into num_heads, and compute attention scores for each head.
    """

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): query tensor. Note that dimension of each query tensor should be the same as key tensor (d_k).
                shape: (batch_size, num_heads, q_seq_len, d_k)
            k (torch.Tensor): key tensor
                shape: (batch_size, num_heads, k_seq_len, d_k)
            v (torch.Tensor): value tensor. Note that sequence length of value tensor should be the same as key tensor (k_seq_len).
                shape: (batch_size, num_heads, k_seq_len, d_v)
            mask (torch.Tensor): mask tensor. If mask[batch, 0, i, j] is True, attention score between i-th token and j-th token is 1e-9.
                shape: (batch_size, 1, q_seq_len, k_seq_len)
                default: None

        Returns:
            context (torch.Tensor): context vector
                shape: (batch_size, num_heads, q_seq_len, d_v)
            attention (torch.Tensor): attention weights
                shape: (batch_size, num_heads, q_seq_len, k_seq_len)
        """

        d_k = k.size(3)
        # Compute attention score
        k_t = k.transpose(2, 3)   # shape: (batch_size, num_heads, d_k, k_seq_len)
        score = torch.matmul(q, k_t) / (d_k ** 0.5)    # shape: (batch_size, num_heads, q_seq_len, k_seq_len)

        # Apply mask
        if mask is not None:
            score = score.masked_fill(mask == 0, 1e-9)

        # Compute attention weights
        attention = self.softmax(score)    # shape: (batch_size, num_heads, q_seq_len, k_seq_len), sum(dim=-1) = 1

        # Compute context vector
        context = torch.matmul(attention, v)    # shape: (batch_size, num_heads, q_seq_len, d_v)

        return context, attention
    

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer
    Receive Q, K, V tensors and compute multi-head attention scores
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model (int): dimension of model
            num_heads (int): number of heads
                d_model should be divisible by num_heads
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads   # confine to d_k == d_v
        # input Q, K, V should have dimension of d_model
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): query tensor
                shape: (batch_size, q_seq_len, d_model)
            k (torch.Tensor): key tensor
                shape: (batch_size, k_seq_len, d_model)
            v (torch.Tensor): value tensor
                shape: (batch_size, k_seq_len, d_model)
            mask (torch.Tensor): mask tensor. If mask[batch, i, j] is 0, attention score between i-th token and j-th token is -inf.
                shape: (batch_size, q_seq_len, k_seq_len)
                default: None

        Returns:
            context (torch.Tensor): context vector
                shape: (batch_size, q_seq_len, d_model)
            attention (torch.Tensor): attention weights
                shape: (batch_size, num_heads, q_seq_len, k_seq_len)
        """

        batch_size, q_seq_len, _ = q.size()
        k_seq_len = k.size(1)
        assert k.size() == v.size()

        # Linear transformation
        q = self.W_q(q)   # shape: (batch_size, q_seq_len, d_model)
        q = q.view(batch_size, q_seq_len, self.num_heads, self.d_k).transpose(1, 2)   # shape: (batch_size, num_heads, q_seq_len, d_k)
        k = self.W_k(k)
        k = k.view(batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)   # shape: (batch_size, num_heads, k_seq_len, d_k)
        v = self.W_v(v)
        v = v.view(batch_size, k_seq_len, self.num_heads, self.d_v).transpose(1, 2)   # shape: (batch_size, num_heads, k_seq_len, d_v)

        context, attention = self.attention(q, k, v, mask)

        # Concatenate multi-heads
        context = context.transpose(1, 2).contiguous().view(batch_size, q_seq_len, self.d_v * self.num_heads)   # shape: (batch_size, q_seq_len, d_model)

        # Linear transformation
        output = self.linear(context)    # shape: (batch_size, q_seq_len, d_model)

        return output, attention
