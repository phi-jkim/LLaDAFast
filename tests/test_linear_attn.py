import torch
from llada_fast.modeling.linear_attention import OrderInvariantKernelLinearAttention

class MockConfig:
    def __init__(self, hidden_size=128, num_attention_heads=4):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

def test_linear_attention_shape():
    config = MockConfig()
    block_size = 32
    seq_len = 100
    batch_size = 2
    
    attn = OrderInvariantKernelLinearAttention(config, block_size=block_size)
    
    q = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    k = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    v = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    
    output = attn(q, k, v)
    
    assert output.shape == (batch_size, config.num_attention_heads, seq_len, config.head_dim)

def test_linear_attention_bidirectional_property():
    """
    In a block-bidirectional attention, changing a token at index j 
    should affect the output at index i within the same block.
    """
    config = MockConfig()
    block_size = 32
    seq_len = 64
    batch_size = 1
    
    attn = OrderInvariantKernelLinearAttention(config, block_size=block_size)
    
    q = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    k = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    v = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    
    # Original output
    out1 = attn(q, k, v)
    
    # Modify v at index 10 (block 0)
    v_mod = v.clone()
    v_mod[:, :, 10, :] += 1.0
    
    out2 = attn(q, k, v_mod)
    
    # Output at index 5 (block 0) should change
    assert not torch.allclose(out1[:, :, 5, :], out2[:, :, 5, :])
    
    # Output at index 40 (block 1) should NOT change
    assert torch.allclose(out1[:, :, 40, :], out2[:, :, 40, :], atol=1e-5)

if __name__ == "__main__":
    test_linear_attention_shape()
    test_linear_attention_bidirectional_property()
    print("All tests passed!")
