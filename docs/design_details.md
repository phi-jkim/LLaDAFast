# LLaDAFast Technical Details

## 1. RoPE Implementation (Standard, No Scaling)
We use the **Standard Rotary Positional Embedding (RoPE)** formula to match the pre-training of LLaDA 2.1-mini:
- **Base ($\theta$)**: `600,000` (Crucial for 32K context).
- **Scaling**: None (`rope_scaling: null`).
- **Implementation**: We manually calculate `inv_freq` in the modeling code to avoid versioning conflicts with the Transformers library's scaling functions. This ensures exact mathematical equivalence to the Teacher model.

## 2. Block-Bidirectional Linear Attention

### Is it Bidirectional?
**Yes**, but with a "block-level" constraint. In standard softmax attention, bidirectional means every token $i$ can see every token $j$. In LLaDAFast, we divide the sequence into blocks (e.g., 512 tokens). Within each block, the attention is fully bidirectional.

### Implementation & Efficiency
We use a kernel-based linear attention:
1. **Feature Map**: Queries and Keys are transformed via a learnable feature map $\phi(x)$. We use $\text{ELU}(x) + 1$ to ensure positivity.
   $$\phi(x) = \text{ELU}(x \cdot W_{scale} + W_{bias}) + 1.0$$
2. **One-Time Block Computation (The Speedup)**: For each block, we compute global "state" tensors **exactly once**:
   - $S = \sum_{j \in block} \phi(k_j) v_j^\top$ (The "Context" matrix, $d \times d$)
   - $Z = \sum_{j \in block} \phi(k_j)$ (The "Normalization" vector, $d$)
3. **Retrieval**: Every query in the block retrieves from these same pre-computed $S$ and $Z$ tensors:
   - $o_i = (\phi(q_i)^\top S) / (\phi(q_i)^\top Z)$

**This is the core speedup**: Instead of chaque query attending to all keys individually ($O(N^2)$), all queries in a block share a single compressed representation of the keys/values.

## 2. LLaDA 2.1: Blockwise Diffusion vs Autoregressive

### Does it do blockwise autoregressive?
**No.** LLaDA 2.1 is fundamentally a **Diffusion Model (Non-Autoregressive)**, but it uses a unique **Draft-and-Edit** paradigm.

In traditional "AR blockwise" (like Speculative Decoding or BPD), you predict multiple future tokens and verify them. Once they are verified, they are "locked in" and never change.

**LLaDA 2.1 Token Editing** is more powerful:
- **T2T (Token-to-Token)**: It can revise tokens that it already generated in previous steps.
- **Bi-Directional Correction**: If the model determines that a token in "Block 1" is inconsistent with the newly generated "Block 2", it can use its bidirectional attention to re-edit Block 1.
- **M2T + T2T**: It simultaneously unmasks new tokens (Mask-to-Token) and corrects existing ones (Token-to-Token).

So, while it generates content in chunks (blocks), it is not "Autoregressive" in the traditional sense, but rather a **multi-block iterative refinement** process.

### The Verdict: Is it AR?
It is **Block-wise AR**. 
- It uses a **Causal Mask** between blocks to allow for **KV Caching** (speed).
- It uses **Bidirectional Attention** inside each block to allow for **Diffusion/Editing** (quality).

### How Distillation Happens (Step 1)
The primary goal is to make the **Linear Attention (Student)** behave exactly like the **Softmax Attention (Teacher)** at the representation level.

- **Loss Metric**: We use **Layer-wise Hidden State MSE**. 
- **Comparison**: We compare the output hidden states of each transformer block in the Student against the corresponding block in the Teacher.
- **Why not AR Testing?**: Distillation happens on a fixed sequence (Teacher's prompt). We don't need to "generate" tokens during this step; we just need the Student's "understanding" (hidden states) to match the Teacher's.
- **Frozen Teacher**: The Teacher provides a "perfect" reference. The Student's Linear kernels ($\phi$) are optimized to minimize the MSE, effectively learning a linear approximation of the complex softmax attention scores.

#### Clean Past, Noisy Present
This hybrid approach works because:
1. **The Past is Fixed**: Once a block is fully denoised, it is added to the KV Cache. It is now "clean" and acts as a static prefix for all future blocks.
2. **The Present is Fluid**: Inside the current 512-token block, the model can edit and refine using bidirectional attention.
3. **The Global Context**: Newer blocks attend to fully denoised "clean" versions of previous blocks, preventing the compounding noise issues that plague traditional diffusion models.

### Default Block Size
While 512 is not a "hard-coded" default for the base LLaDA model (which supports 32K context), it is an industry-standard choice for **Block-Linear** implementations to balance memory savings vs complexity. The official LLaDA 2.1 samplers use variable `block_length` depending on the speed/quality tradeoff (S Mode vs Q Mode).

## 3. KV Cache in Linear Attention

### How it maintains KV Cache
In standard Transformers, the KV cache grows linearly with sequence length ($O(N)$), which is why 32K context usually requires massive VRAM or techniques like FlashAttention/GQA.

In our **Linear LLaDAFast**:
- The "KV Cache" is simply the $S$ and $Z$ tensors.
- These tensors have a **fixed size** ($d \times d$) regardless of whether the block is 128 or 32,768 tokens long.
- As the model processes new blocks (or refines existing ones), it only needs to keep these "summary" matrices in memory, drastically reducing the memory footprint for long-context generation.

### How it Works
- **Fixed Targets**: The original LLaDA 2.1-mini (Teacher) is frozen (no-grad). It provides the "gold standard" for what the hidden states should look like after a softmax attention block.
- **Student Learning**: The student model has the same weights but replaces Softmax with our Linear Attention.
- **Loss**: We minimize the Mean Squared Error (MSE) between the Teacher's hidden states and the Student's hidden states.
- **Why**: This "teaches" the linear kernel parameters $\phi$ to produce outputs that match the complex softmax routing as closely as possible.

## 3. Step 2: LoRA Recovery

### How it Works
Linearization often causes a small drop in "perceptual" quality or reasoning. To fix this, we use **LoRA (Low-Rank Adaptation)**.
- **Parameter Efficient**: We only train small rank-8 matrices added to existing linear projections ($W_q, W_k, W_v, W_o$).
- **Diffusion Objective**: We train using LLaDA's denoising loss. We randomly mask tokens in UltraChat dialogues and ask the model to predict the original tokens.
- **Result**: LoRA "warps" the weights to work optimally with the new linear attention mechanism, recovering the benchmark performance lost during Step 1.
