import torch
import torch.nn as nn
from llada_fast.modeling.linear_attention import OrderInvariantKernelLinearAttention

class MockConfig:
    def __init__(self, hidden_size=128, num_attention_heads=4, feature_dim=64):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.feature_dim = feature_dim

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
    Within-block: bidirectional (changing v[j] changes output at i in same block).
    Across-block: CAUSAL state — block 1 sees block 0 context, but block 0 does NOT
    see block 1 (future leakage check).
    """
    config = MockConfig()
    block_size = 32
    seq_len = 64
    batch_size = 1

    attn = OrderInvariantKernelLinearAttention(config, block_size=block_size)

    q = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    k = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)
    v = torch.randn(batch_size, config.num_attention_heads, seq_len, config.head_dim)

    out1 = attn(q, k, v)

    # Modify v at index 10 (block 0)
    v_mod = v.clone()
    v_mod[:, :, 10, :] += 1.0
    out2 = attn(q, k, v_mod)

    # Same block: pos 5 should change when pos 10 (block 0) changes
    assert not torch.allclose(out1[:, :, 5, :], out2[:, :, 5, :], atol=1e-5)

    # Causal: block 0 (pos 5) must NOT be affected by a change in block 1 (pos 40)
    v_mod2 = v.clone()
    v_mod2[:, :, 40, :] += 1.0
    out3 = attn(q, k, v_mod2)
    assert torch.allclose(out1[:, :, 5, :], out3[:, :, 5, :], atol=1e-5), \
        "Block 0 output must not change when only block 1 v changes (causal)"

def test_hedgehog_gradient():
    """
    Verify that gradients flow back to the hedgehog_weights.
    """
    config = MockConfig()
    attn = OrderInvariantKernelLinearAttention(config)
    
    q = torch.randn(1, config.num_attention_heads, 32, config.head_dim, requires_grad=True)
    k = torch.randn(1, config.num_attention_heads, 32, config.head_dim, requires_grad=True)
    v = torch.randn(1, config.num_attention_heads, 32, config.head_dim, requires_grad=True)
    
    output = attn(q, k, v)
    loss = output.sum()
    loss.backward()
    
    assert attn.hedgehog_weights.grad is not None
    assert not torch.allclose(attn.hedgehog_weights.grad, torch.zeros_like(attn.hedgehog_weights.grad))

if __name__ == "__main__":
    test_linear_attention_shape()
    test_linear_attention_bidirectional_property()
    test_hedgehog_gradient()
    print("All tests passed!")
