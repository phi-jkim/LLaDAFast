import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderInvariantKernelLinearAttention(nn.Module):
    """
    Order-invariant kernel linear attention that remains bidirectional within a block.
    
    Args:
        config: Model configuration containing hidden_size, num_heads, etc.
        block_size: Size of the block for bidirectional attention.
    """
    def __init__(self, config, block_size=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = block_size or getattr(config, "block_size", 512)
        
        # Learnable feature map parameters
        self.phi_scale = nn.Parameter(torch.ones(self.num_heads, 1, self.head_dim))
        self.phi_bias = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def feature_map(self, x):
        """
        Applies the learnable feature map phi.
        x: (batch, num_heads, num_blocks, block_size, head_dim)
        """
        x = x * self.phi_scale.unsqueeze(1) + self.phi_bias.unsqueeze(1)
        return F.elu(x) + 1.0

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Forward pass for block-bidirectional linear attention.
        query_states, key_states, value_states: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Pad seq_len to be multiple of block_size if necessary
        pad_len = num_blocks * self.block_size - seq_len
        if pad_len > 0:
            query_states = F.pad(query_states, (0, 0, 0, pad_len))
            key_states = F.pad(key_states, (0, 0, 0, pad_len))
            value_states = F.pad(value_states, (0, 0, 0, pad_len))

        # Reshape into blocks: (batch, num_heads, num_blocks, block_size, head_dim)
        q = query_states.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)
        k = key_states.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)
        v = value_states.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)

        # Apply feature map
        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)

        # S = sum_{j in block} phi(k_j) v_j^T -> (batch, num_heads, num_blocks, head_dim, head_dim)
        S = torch.einsum('b h n s d, b h n s m -> b h n d m', phi_k, v)

        # Z = sum_{j in block} phi(k_j) -> (batch, num_heads, num_blocks, head_dim)
        Z = phi_k.sum(dim=-2)

        # o_i = (phi(q_i)^T S) / (phi(q_i)^T Z)
        num = torch.einsum('b h n s d, b h n d m -> b h n s m', phi_q, S)
        
        # den = phi_q^T Z -> (batch, num_heads, num_blocks, block_size)
        den = torch.einsum('b h n s d, b h n d -> b h n s', phi_q, Z)
        den = den.unsqueeze(-1) + 1e-6

        output = num / den

        # Reshape back to original sequence length
        output = output.view(batch_size, num_heads, -1, head_dim)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :, :seq_len, :]

        return output
