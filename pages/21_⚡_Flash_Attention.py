import streamlit as st

st.title("⚡ Topic 21: Flash Attention")

st.markdown("""
---

## 🎯 The Speed Revolution: 2-4x Faster Attention

Imagine training GPT-3 in **half the time** with the **same hardware**. Or generating text **4x faster**. Or handling **128k token contexts** that were previously impossible.

**This isn't hypothetical - Flash Attention makes it real!**

**Flash Attention** is one of the most important algorithmic innovations in modern deep learning. It's not a new architecture - it's a **smarter way to compute the SAME attention**, exploiting GPU hardware to be dramatically faster.

**Impact:**
- 🚀 **2-4x faster** than standard attention
- 🚀 **10-20x less memory** usage
- 🚀 **Exact same output** (not an approximation!)
- 🚀 Used in **ALL modern LLM training** (GPT-4, Claude 3, Gemini, LLaMA)

**This topic is different**: We won't write CUDA kernels (that's expert-level). Instead, we'll understand:
1. **Why** standard attention is slow
2. **What** Flash Attention does differently
3. **How** it achieves massive speedups
4. **When** to use it (spoiler: always!)

---

## 🐌 The Problem: Attention is Memory-Bound

### Standard Attention Implementation

```python
def standard_attention(Q, K, V):
    # Q, K, V: [batch, num_heads, seq_len, head_dim]

    # Step 1: Compute attention scores
    scores = Q @ K.T / sqrt(d_k)  # [batch, heads, seq_len, seq_len]

    # Step 2: Apply softmax
    attn_weights = softmax(scores)  # [batch, heads, seq_len, seq_len]

    # Step 3: Apply to values
    output = attn_weights @ V  # [batch, heads, seq_len, head_dim]

    return output
```

**Looks simple, right? What's the problem?**

### The Hidden Bottleneck: Memory Hierarchy

Modern GPUs have a **memory hierarchy**:

```
┌─────────────────────────────────────┐
│ SRAM (On-Chip) - 20 MB              │  ← VERY FAST (19 TB/s)
│ "Registers + Shared Memory + Cache" │  ← But TINY!
└─────────────────────────────────────┘
              ↕ (slow data movement)
┌─────────────────────────────────────┐
│ HBM (High Bandwidth Memory) - 40 GB │  ← SLOWER (1.5 TB/s)
│ "Main GPU Memory (VRAM)"            │  ← But LARGE
└─────────────────────────────────────┘
```

**The problem:** Standard attention makes **MANY trips** between SRAM and HBM!

### Memory Access Pattern (Standard Attention)

```python
# Step 1: Load Q and K from HBM → Compute scores → Write scores to HBM
scores = Q @ K.T  # Trip 1: Read Q, K | Trip 2: Write scores

# Step 2: Load scores from HBM → Apply softmax → Write result to HBM
attn = softmax(scores)  # Trip 3: Read scores | Trip 4: Write attn

# Step 3: Load attn and V from HBM → Compute output → Write to HBM
output = attn @ V  # Trip 5: Read attn, V | Trip 6: Write output
```

**6 HBM accesses!** Each is SLOW (1.5 TB/s vs 19 TB/s for SRAM).

### The Bottleneck: O(N²) Attention Matrix

```python
# For sequence length N = 4096
seq_len = 4096
scores = Q @ K.T  # Shape: [4096, 4096] = 16M elements!

# With 32 heads and fp16 (2 bytes):
memory = 4096 * 4096 * 32 * 2 = 1 GB just for attention scores!
```

**Problems:**
1. **Huge intermediate tensor** (N² for scores)
2. **Slow HBM reads/writes** dominate computation time
3. **Memory-bound, not compute-bound** - GPU sits idle waiting for data!

**Attention is quadratic in MEMORY, not just computation!**

---

## 💡 Flash Attention: The Brilliant Solution

### Core Insight: Fuse Operations + Tile Processing

**Key ideas:**
1. **Don't materialize the full N² attention matrix**
2. **Process in tiles** that fit in fast SRAM
3. **Fuse operations** (compute + softmax + output in one kernel)
4. **Recompute instead of storing** intermediate values

**Result:** Dramatically fewer HBM accesses!

### The Magic: Tiling

Instead of computing the full attention matrix at once:

```python
# Standard: Compute full N×N matrix at once
scores = Q @ K.T  # [N, N] - doesn't fit in SRAM!

# Flash Attention: Process in blocks
for block_i in Q_blocks:
    for block_j in K_blocks:
        # Compute attention for this small tile
        # Tile fits in fast SRAM!
        scores_tile = Q_block_i @ K_block_j.T
        # Immediately use it, don't store
```

**Tiles are small** (e.g., 128×128) → fit in SRAM → fast!

### Softmax Trick: Online Softmax

**Challenge:** Softmax needs the full row to normalize!

```python
# Standard softmax
softmax(x) = exp(x) / sum(exp(x))  # Needs full row!
```

**Flash Attention solution:** Incremental softmax update!

```python
# Process row in chunks, update running statistics
def online_softmax(chunks):
    m = -inf  # Running max
    d = 0     # Running sum

    for chunk in chunks:
        m_new = max(m, max(chunk))
        d_new = d * exp(m - m_new) + sum(exp(chunk - m_new))
        m, d = m_new, d_new

    # Can compute softmax incrementally!
```

**This allows processing attention in blocks without materializing the full matrix!**

---

## 🔬 Flash Attention Algorithm (High-Level)

### Conceptual Algorithm

```python
def flash_attention(Q, K, V, block_size=128):
    """
    Flash Attention - Conceptual (actual implementation is in CUDA)

    Key ideas:
    1. Split Q, K, V into blocks
    2. Process blocks in SRAM (fast memory)
    3. Incrementally compute attention output
    4. Never materialize full attention matrix
    """
    N = Q.shape[0]  # Sequence length
    d = Q.shape[1]  # Head dimension

    # Split into blocks
    num_blocks = N // block_size

    # Initialize output and statistics
    O = zeros(N, d)  # Output
    l = zeros(N)     # Softmax denominator (per row)
    m = -inf * ones(N)  # Softmax max (per row)

    # Outer loop: Query blocks
    for i in range(num_blocks):
        # Load Q block into SRAM
        Q_i = Q[i*block_size : (i+1)*block_size]  # [block_size, d]

        # Initialize block output
        O_i = zeros(block_size, d)
        l_i = zeros(block_size)
        m_i = -inf * ones(block_size)

        # Inner loop: Key/Value blocks
        for j in range(num_blocks):
            # Load K, V blocks into SRAM
            K_j = K[j*block_size : (j+1)*block_size]
            V_j = V[j*block_size : (j+1)*block_size]

            # Compute attention scores (in SRAM!)
            S_ij = Q_i @ K_j.T / sqrt(d)  # [block_size, block_size]

            # Update statistics and output (incremental softmax)
            m_i_new = max(m_i, max(S_ij, dim=1))

            # Attention weights for this block
            P_ij = exp(S_ij - m_i_new)

            # Update running sum
            l_i_new = exp(m_i - m_i_new) * l_i + sum(P_ij, dim=1)

            # Update output (incremental)
            O_i = (l_i / l_i_new) * exp(m_i - m_i_new) * O_i + (1/l_i_new) * P_ij @ V_j

            # Update statistics
            l_i = l_i_new
            m_i = m_i_new

        # Write block output to HBM
        O[i*block_size : (i+1)*block_size] = O_i

    return O
```

**Key differences from standard attention:**
- ✅ Blocks fit in SRAM (fast)
- ✅ Attention scores computed in small tiles
- ✅ Never store full N² matrix
- ✅ Fewer HBM accesses
- ✅ **Exact same result!**

---

## 📊 Performance Improvements

### Speed Comparison

```
Sequence Length: 2048 tokens
Hardware: A100 GPU

Standard Attention:
- Forward pass: 12.5 ms
- Backward pass: 32.1 ms
- Total: 44.6 ms

Flash Attention:
- Forward pass: 3.2 ms (3.9x faster!)
- Backward pass: 8.7 ms (3.7x faster!)
- Total: 11.9 ms (3.7x faster overall!)
```

**Speedup increases with sequence length!**

```
Sequence Length: 4096 tokens

Standard Attention: 178 ms
Flash Attention: 47 ms (3.8x faster!)

Sequence Length: 8192 tokens

Standard Attention: 712 ms
Flash Attention: 189 ms (3.8x faster!)
```

### Memory Comparison

```
Sequence Length: 4096
Batch Size: 8
Heads: 32
Head Dim: 128

Standard Attention:
- Attention scores: 4096 × 4096 × 32 × 8 × 2 bytes = 8.6 GB
- Peak memory: ~12 GB

Flash Attention:
- No full attention matrix stored!
- Peak memory: ~0.8 GB (15x reduction!)
```

**Memory savings are HUGE for long sequences!**

---

## 🚀 Flash Attention 2: Even Faster

In 2023, **Flash Attention 2** was released with further optimizations:

**Improvements:**
1. **Better parallelism**: Optimized work distribution across GPU cores
2. **Reduced non-matmul operations**: Minimize overhead
3. **Better handling of attention masks**: Causal masking optimized

**Results:**
- **2x faster** than Flash Attention 1
- **4-8x faster** than standard attention
- **Same memory benefits**

### Performance (Flash Attention 2)

```
Sequence Length: 2048 tokens
Hardware: A100 GPU

Standard Attention: 44.6 ms
Flash Attention 1: 11.9 ms (3.7x faster)
Flash Attention 2: 6.1 ms (7.3x faster!)
```

---

## 💻 Using Flash Attention in PyTorch

### PyTorch 2.0+ Built-in Support

Good news! PyTorch 2.0+ has Flash Attention built-in:

```python
import torch
import torch.nn.functional as F

# Modern PyTorch (2.0+)
def efficient_attention(Q, K, V, is_causal=False):
    """
    Uses Flash Attention automatically if available!

    Args:
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        is_causal: Use causal masking (for GPT-style models)

    Returns:
        Output: [batch, num_heads, seq_len, head_dim]
    """
    # PyTorch automatically uses Flash Attention if:
    # 1. CUDA GPU
    # 2. No explicit mask (or causal mask)
    # 3. Supported data type (fp16, bf16)

    output = F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=is_causal,
        dropout_p=0.0
    )

    return output


# Example usage
batch_size = 2
num_heads = 8
seq_len = 2048
head_dim = 64

Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

# This automatically uses Flash Attention!
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print(f"Input shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print("✓ Using Flash Attention automatically!")
```

### External Flash Attention Library

For more control, use the official library:

```python
# Install: pip install flash-attn

from flash_attn import flash_attn_func

def flash_attention_explicit(Q, K, V, causal=False):
    """
    Explicit Flash Attention usage

    Args:
        Q, K, V: [batch, seq_len, num_heads, head_dim]
                 (Note: different layout than PyTorch!)
        causal: Use causal masking

    Returns:
        Output: [batch, seq_len, num_heads, head_dim]
    """
    # Flash Attention expects [batch, seq_len, num_heads, head_dim]
    output = flash_attn_func(Q, K, V, causal=causal)

    return output


# Example (note the different shape!)
Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
K = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
V = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)

output = flash_attn_func(Q, K, V, causal=True)

print(f"Flash Attention output shape: {output.shape}")
```

### Integration in Transformer

```python
import torch.nn as nn
import torch.nn.functional as F

class FlashMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using Flash Attention

    Automatically faster on modern GPUs!
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

    def forward(self, x, is_causal=False):
        batch_size, seq_len, _ = x.shape

        # Project and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Use Flash Attention! (PyTorch 2.0+)
        output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


# Example
model = FlashMultiHeadAttention(d_model=512, num_heads=8).cuda()
x = torch.randn(2, 1024, 512, device='cuda', dtype=torch.float16)

# Fast inference!
with torch.no_grad():
    output = model(x, is_causal=True)

print(f"Output shape: {output.shape}")
print("✓ Using Flash Attention for 3-4x speedup!")
```

---

## 🌍 Real-World Impact

### Training Large Models

```
GPT-3 (175B parameters) training:

Without Flash Attention:
- Training time: ~6 weeks
- GPU memory per sample: 32 GB
- Max sequence length: 2048

With Flash Attention:
- Training time: ~3 weeks (2x faster!)
- GPU memory per sample: 12 GB (2.7x less)
- Max sequence length: 8192 (4x longer!)

Savings: $1M+ in compute costs!
```

### Long Context Models

```
LLaMA 2 with 32k context:

Without Flash Attention:
- Out of memory on A100 (80GB)

With Flash Attention:
- Fits comfortably!
- Can even handle 128k context
- Enables new use cases (full document understanding)
```

### Production Inference

```
Serving ChatGPT-scale model:

Without Flash Attention:
- 150 tokens/second per user
- 4 concurrent users per GPU

With Flash Attention:
- 500 tokens/second per user (3.3x faster!)
- 12 concurrent users per GPU (3x more!)

ROI: 3x more users per GPU = 3x revenue per dollar!
```

---

## 🎓 Key Takeaways

1. **Problem**: Standard attention is memory-bound (too many slow HBM accesses)

2. **Solution**: Flash Attention uses tiling + kernel fusion
   - Processes attention in blocks that fit in fast SRAM
   - Never materializes full N² attention matrix
   - Incremental softmax computation

3. **Performance**: 2-4x faster, 10-20x less memory
   - Flash Attention 2: Up to 8x faster!

4. **Exact**: Same output as standard attention (not approximate)

5. **Easy to use**: Built into PyTorch 2.0+
   ```python
   F.scaled_dot_product_attention(Q, K, V, is_causal=True)
   ```

6. **Universal adoption**: Used in ALL modern LLM training
   - GPT-4, Claude 3, Gemini, LLaMA, Mistral

7. **Enables long contexts**: 32k, 128k token sequences now feasible

**Flash Attention is THE reason modern LLMs can train faster and handle longer contexts!**

---

## 📝 Quiz Time!

Test your understanding of Flash Attention.
""")

# Quiz questions
with st.expander("Question 1: What makes standard attention slow?"):
    st.markdown("""
    **Question**: What is the PRIMARY bottleneck in standard attention that Flash Attention solves?

    A) Slow matrix multiplication
    B) Too many reads/writes to slow GPU memory (HBM) for the large N² attention matrix
    C) Inefficient softmax computation
    D) Too many attention heads
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) Too many reads/writes to slow GPU memory (HBM) for the large N² attention matrix**

        **Explanation**:

        **The bottleneck:**
        ```python
        # Standard attention creates N×N matrix
        scores = Q @ K.T  # [seq_len, seq_len] - HUGE!

        # For seq_len=4096: 16M elements!
        # Stored in slow HBM (1.5 TB/s)
        # Many read/write operations to HBM
        ```

        **Why it's slow:**
        - HBM (main GPU memory): 1.5 TB/s
        - SRAM (on-chip memory): 19 TB/s (12x faster!)
        - N² attention matrix doesn't fit in fast SRAM
        - GPU spends most time waiting for data, not computing

        **Flash Attention solution:**
        - Process in small tiles that fit in SRAM
        - Drastically fewer HBM accesses
        - 2-4x speedup!

        Modern GPUs are memory-bound, not compute-bound for attention!
        """)

with st.expander("Question 2: How does Flash Attention work?"):
    st.markdown("""
    **Question**: What is the core technique Flash Attention uses to achieve speedup?

    A) Approximate attention with fewer computations
    B) Process attention in tiles that fit in fast SRAM and use incremental softmax
    C) Use fewer attention heads
    D) Skip some tokens
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) Process attention in tiles that fit in fast SRAM and use incremental softmax**

        **Explanation**:

        **Key techniques:**

        1. **Tiling:**
        ```python
        # Instead of full N×N matrix
        for Q_block in Q_tiles:  # Small blocks (128×128)
            for K_block in K_tiles:
                # Process tile in fast SRAM!
                scores_tile = Q_block @ K_block.T
        ```

        2. **Incremental softmax:**
        ```python
        # Update softmax statistics incrementally
        # Don't need full row at once!
        m = running_max
        d = running_sum
        # Update as we process blocks
        ```

        3. **Kernel fusion:**
        - Compute attention + softmax + output in one CUDA kernel
        - No intermediate writes to HBM

        **Result:**
        - Same exact output as standard attention
        - Much faster (fewer HBM accesses)
        - Much less memory (no full N² matrix)

        It's an algorithmic + hardware optimization, not an approximation!
        """)

with st.expander("Question 3: Flash Attention accuracy"):
    st.markdown("""
    **Question**: How does Flash Attention's output compare to standard attention?

    A) Approximate - slight quality loss
    B) Exact same output (mathematically equivalent)
    C) Better quality due to numerical stability
    D) Only works for short sequences
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) Exact same output (mathematically equivalent)**

        **Explanation**:

        **Flash Attention is NOT an approximation!**

        ```python
        # These produce IDENTICAL outputs:
        output_standard = standard_attention(Q, K, V)
        output_flash = flash_attention(Q, K, V)

        assert torch.allclose(output_standard, output_flash)  # ✓ True!
        ```

        **How is this possible?**
        - Same computation, just reordered
        - Incremental softmax is mathematically equivalent to full softmax
        - No approximations, no dropped calculations
        - Purely a memory access optimization

        **Benefits:**
        - Drop-in replacement (no retraining needed)
        - Exact numerical equivalence
        - All the speed, none of the quality trade-off

        **Contrast with other methods:**
        - Sparse attention: Approximate (skips some pairs)
        - Linear attention: Approximate (different attention mechanism)
        - Flash Attention: **Exact** (just faster!)
        """)

with st.expander("Question 4: Performance gains"):
    st.markdown("""
    **Question**: What kind of speedup does Flash Attention 2 typically achieve over standard attention?

    A) 10-20% faster
    B) 2x faster
    C) 4-8x faster
    D) 100x faster
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: C) 4-8x faster**

        **Explanation**:

        **Speedup depends on sequence length:**

        ```
        Sequence Length: 1024
        Flash Attention 1: 2-3x faster
        Flash Attention 2: 4-5x faster

        Sequence Length: 2048
        Flash Attention 1: 3-4x faster
        Flash Attention 2: 5-7x faster

        Sequence Length: 4096+
        Flash Attention 1: 4-5x faster
        Flash Attention 2: 6-8x faster
        ```

        **Why speedup increases with length?**
        - Longer sequences = larger N² matrix
        - More HBM accesses in standard attention
        - More benefit from tiling

        **Plus memory savings:**
        - 10-20x less peak memory
        - Enables sequences that couldn't fit before

        **Real impact:**
        - GPT-3 training: ~2x overall speedup
        - Enables 128k context windows
        - Millions saved in compute costs
        """)

with st.expander("Question 5: Using Flash Attention"):
    st.markdown("""
    **Question**: How can you use Flash Attention in PyTorch 2.0+?

    A) Need to write custom CUDA kernels
    B) Use F.scaled_dot_product_attention() - it automatically uses Flash Attention
    C) Need to install separate library and rewrite model
    D) Only available in TensorFlow
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: B) Use F.scaled_dot_product_attention() - it automatically uses Flash Attention**

        **Explanation**:

        **Super easy in PyTorch 2.0+:**

        ```python
        import torch.nn.functional as F

        # Automatically uses Flash Attention!
        output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True  # For GPT-style models
        )
        ```

        **Requirements:**
        - PyTorch 2.0+
        - CUDA GPU
        - FP16 or BF16 precision
        - No explicit attention mask (or causal mask)

        **When it's used:**
        - PyTorch checks if Flash Attention is available
        - Automatically dispatches to Flash Attention kernel
        - Falls back to standard if not available
        - Completely transparent!

        **Alternative:**
        ```python
        # Explicit library (more control)
        from flash_attn import flash_attn_func
        output = flash_attn_func(Q, K, V, causal=True)
        ```

        **Best practice:** Use PyTorch's built-in - it's automatic and well-tested!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand Flash Attention - the algorithmic breakthrough that makes modern LLMs feasible!

But there's one more critical optimization for inference:

Next topics:
- **Topic 22**: KV Cache & Efficient Inference - the key to fast autoregressive generation
- **Topic 23**: Mixture of Experts (MoE) - scaling to trillions of parameters

**You now understand the optimization that powers EVERY modern LLM training run!** 🚀

---

*Navigation: ← Previous: Topic 20 (Grouped Query Attention) | Next: Topic 22 (KV Cache) →*
""")
