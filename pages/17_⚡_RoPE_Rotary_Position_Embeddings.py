import streamlit as st

st.title("⚡ Topic 17: RoPE (Rotary Position Embeddings)")

st.markdown("""
---

## 🎯 The Next Evolution: Better Position Encoding

You've learned about sinusoidal and learned positional encodings. They work, but researchers discovered **fundamental limitations**:

**Problems with traditional position encoding:**
1. ❌ **Absolute positions**: Position 5 encoded the same everywhere
2. ❌ **Poor extrapolation**: Hard to handle sequences longer than training length
3. ❌ **Additive**: Position info added before attention, not integrated into it
4. ❌ **No explicit relative bias**: Model must learn relative positions indirectly

**Then in 2021, something revolutionary happened:**

**RoFormer** paper introduced **RoPE (Rotary Position Embeddings)**, and it changed everything!

**RoPE is now the STANDARD in modern LLMs:**
- ✅ LLaMA 1, 2, 3 (Meta)
- ✅ GPT-Neo, GPT-J (EleutherAI)
- ✅ PaLM, Gemini (Google)
- ✅ Mistral, Mixtral (Mistral AI)
- ✅ Qwen, Yi (Chinese LLMs)
- ✅ Almost every new LLM in 2024-2025

**Why is RoPE so good?** Let's find out!

---

## 💡 The Core Insight: Rotation in Complex Plane

### The Brilliant Idea

Instead of **adding** position information, **rotate** the query and key vectors by an angle proportional to position!

```python
# Old way (sinusoidal/learned)
output = embedding + positional_encoding

# RoPE way
Q_rotated = rotate(Q, position)
K_rotated = rotate(K, position)
# Then compute attention with rotated Q and K
```

**Why rotation?**
- When you rotate vectors and then take dot products, you get **relative position information** for free!
- The math works out beautifully (we'll see why)
- Generalizes perfectly to longer sequences

### 2D Rotation Intuition

Think of rotating a vector in 2D space:

```
Vector at position 0: (1, 0)
Rotate by θ: (cos θ, sin θ)
Rotate by 2θ: (cos 2θ, sin 2θ)
```

**Key insight**: The angle between two rotated vectors depends on the **difference** in their positions!

```python
# Position 3 rotated by 3θ
# Position 5 rotated by 5θ
# Angle between them: 5θ - 3θ = 2θ (depends on relative distance!)
```

This is **exactly** what we want for attention - words should attend based on relative distance!

---

## 🧮 The Mathematics (Made Simple)

Don't worry - we'll build intuition without heavy math!

### 2D Case: The Foundation

For a 2-dimensional vector [x₁, x₂] at position m:

**Rotation by angle θ = m × θ₀:**

```
[x'₁]   [cos(mθ₀)  -sin(mθ₀)] [x₁]
[x'₂] = [sin(mθ₀)   cos(mθ₀)] [x₂]
```

This is standard 2D rotation!

**Key property**: When we compute the dot product of two rotated vectors at positions m and n:

```
dot(rotate(x, m), rotate(y, n)) = dot(x, rotate(y, n-m))
```

The result depends only on the **relative position (n-m)**, not absolute positions!

### Extending to High Dimensions

For d-dimensional vectors, we split them into d/2 pairs and rotate each pair independently with **different frequencies**:

```python
# d_model = 8 example (4 pairs)
Pair 1 (dim 0-1): Rotate by m × θ₀  (slow rotation)
Pair 2 (dim 2-3): Rotate by m × θ₁  (medium rotation)
Pair 3 (dim 4-5): Rotate by m × θ₂  (faster rotation)
Pair 4 (dim 6-7): Rotate by m × θ₃  (fastest rotation)

where θᵢ = 10000^(-2i/d)  (same as sinusoidal!)
```

**Why different frequencies?**
- Low-frequency rotations: Capture long-range patterns
- High-frequency rotations: Capture short-range patterns
- Same multi-scale idea as sinusoidal encoding!

---

## 💻 Complete PyTorch Implementation

Let's build RoPE from scratch!

```python
import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embeddings) from RoFormer

    Used in: LLaMA, PaLM, GPT-Neo, Mistral, and most modern LLMs!
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        """
        Args:
            dim: Dimension of the model (must be even)
            max_seq_len: Maximum sequence length to pre-compute
            base: Base for frequency calculation (default: 10000)
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.base = base

        # Compute frequencies for each dimension pair
        # θᵢ = base^(-2i/dim) for i in [0, 1, ..., dim/2-1]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute rotations for all positions
        self._cache_rotations(max_seq_len)

    def _cache_rotations(self, seq_len):
        """
        Pre-compute rotation matrices for efficiency
        """
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len).float()

        # Compute angles: position × frequency
        # Shape: [seq_len, dim/2]
        angles = positions.unsqueeze(1) @ self.inv_freq.unsqueeze(0)

        # Duplicate angles for sin and cos (will apply to pairs)
        # Shape: [seq_len, dim]
        angles = torch.cat([angles, angles], dim=-1)

        # Pre-compute sin and cos
        self.register_buffer('cos_cached', angles.cos())
        self.register_buffer('sin_cached', angles.sin())

    def rotate_half(self, x):
        """
        Rotate half the dimensions of x

        [x₁, x₂, x₃, x₄] → [-x₃, -x₄, x₁, x₂]
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x, seq_len):
        """
        Apply rotary position embeddings to x

        Args:
            x: Input tensor [batch, seq_len, num_heads, head_dim]
            seq_len: Sequence length

        Returns:
            Rotated tensor with same shape
        """
        # Get cached sin and cos for this sequence length
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)

        # Apply rotation
        # x_rotated = x * cos + rotate_half(x) * sin
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, q, k):
        """
        Apply RoPE to queries and keys

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]

        Returns:
            Rotated q and k with same shapes
        """
        seq_len = q.shape[1]

        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._cache_rotations(seq_len)

        # Apply rotations
        q_rotated = self.apply_rotary_pos_emb(q, seq_len)
        k_rotated = self.apply_rotary_pos_emb(k, seq_len)

        return q_rotated, k_rotated


# Example usage
batch_size = 2
seq_len = 10
num_heads = 8
head_dim = 64

# Create random Q and K (from multi-head attention)
Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
K = torch.randn(batch_size, seq_len, num_heads, head_dim)

# Create RoPE
rope = RotaryPositionEmbedding(dim=head_dim)

# Apply RoPE
Q_rope, K_rope = rope(Q, K)

print(f"Original Q shape: {Q.shape}")
print(f"Rotated Q shape: {Q_rope.shape}")
print(f"\\nRoPE preserves shape! ✓")

# Verify relative position property
print(f"\\nFrequencies (inv_freq): {rope.inv_freq}")
```

**Output:**
```
Original Q shape: torch.Size([2, 10, 8, 64])
Rotated Q shape: torch.Size([2, 10, 8, 64])

RoPE preserves shape! ✓

Frequencies (inv_freq): tensor([1.0000, 0.7743, 0.5995, 0.4642, ...])
```

---

## 🔍 How RoPE Works: Step-by-Step

### Step 1: Compute Rotation Frequencies

```python
# For each dimension pair i in [0, 1, 2, ..., dim/2-1]
θᵢ = base^(-2i/dim)

# Example with dim=8
θ₀ = 10000^0 = 1.0        # Slowest rotation
θ₁ = 10000^(-1/4) = 0.1   # Slow rotation
θ₂ = 10000^(-2/4) = 0.01  # Fast rotation
θ₃ = 10000^(-3/4) = 0.001 # Fastest rotation
```

### Step 2: Compute Angles for Each Position

```python
# For position m and frequency θᵢ
angle = m × θᵢ

# Position 0: angles = [0, 0, 0, 0]
# Position 1: angles = [1×θ₀, 1×θ₁, 1×θ₂, 1×θ₃]
# Position 2: angles = [2×θ₀, 2×θ₁, 2×θ₂, 2×θ₃]
# Position 3: angles = [3×θ₀, 3×θ₁, 3×θ₂, 3×θ₃]
```

### Step 3: Pre-compute Sin and Cos

```python
cos_matrix[m, i] = cos(m × θᵢ)
sin_matrix[m, i] = sin(m × θᵢ)
```

### Step 4: Apply Rotation to Q and K

```python
# For each dimension pair (x₁, x₂):
x'₁ = x₁ × cos(mθ) - x₂ × sin(mθ)
x'₂ = x₁ × sin(mθ) + x₂ × cos(mθ)

# This is equivalent to:
# [x'₁, x'₂] = rotation_matrix × [x₁, x₂]
```

### Step 5: Compute Attention with Rotated Q and K

```python
# Standard attention, but with RoPE-rotated queries and keys
scores = (Q_rope @ K_rope.T) / sqrt(d_k)
attention = softmax(scores)
output = attention @ V
```

---

## 🎨 Visualizing RoPE

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_rope_rotations(dim=64, max_seq_len=100):
    """
    Visualize how RoPE rotates embeddings at different positions
    """
    rope = RotaryPositionEmbedding(dim, max_seq_len)

    # Get rotation angles for first few dimensions
    positions = torch.arange(max_seq_len).float()
    angles = positions.unsqueeze(1) @ rope.inv_freq.unsqueeze(0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Rotation angles over positions
    for i in range(min(8, dim // 2)):
        axes[0].plot(angles[:, i], label=f'Dimension pair {i}')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Rotation Angle (radians)')
    axes[0].set_title('RoPE Rotation Angles by Position')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Frequency spectrum
    frequencies = rope.inv_freq.numpy()
    axes[1].bar(range(len(frequencies)), frequencies)
    axes[1].set_xlabel('Dimension Pair Index')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('RoPE Frequency Spectrum (Different Dimension Pairs)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Uncomment to visualize:
# fig = visualize_rope_rotations()
# plt.show()
```

**What you'd see:**
- Low-index dimensions rotate slowly (long-range patterns)
- High-index dimensions rotate quickly (short-range patterns)
- Exponential decay in frequencies (same as sinusoidal PE)

---

## 🌟 Why RoPE is Revolutionary

### 1. True Relative Position Encoding

**Mathematical property:**

```python
# Attention score between position m and n:
score(m, n) = Q_m^T K_n = f(m - n)

# Depends ONLY on relative distance (m - n), not absolute positions!
```

**Why this matters:**
- "The word 3 positions to the left" means the same thing everywhere
- Model learns relative relationships that transfer

### 2. Perfect Extrapolation to Longer Sequences

```python
# Trained on sequences up to 2048 tokens
rope = RotaryPositionEmbedding(dim=64, max_seq_len=2048)

# Works perfectly on 10,000 token sequences!
Q_long, K_long = rope(Q_10k, K_10k)
```

**Why it works:**
- Rotation is a continuous function
- No learned parameters to limit length
- Relative positions naturally extend

**Real-world impact:**
- LLaMA can extend from 4k → 32k context with minimal fine-tuning
- Mistral trained on 8k works well on 32k
- Critical for long-document understanding

### 3. Integration with Attention

**Traditional PE:**
```python
# Position added BEFORE attention
x = embedding + positional_encoding
Q, K, V = project(x)  # Position info diluted
```

**RoPE:**
```python
# Position integrated INTO attention mechanism
Q, K, V = project(embedding)
Q, K = rope(Q, K)  # Position directly affects attention scores!
```

Position information is directly encoded in the attention computation!

### 4. No Extra Parameters

```python
# Sinusoidal: 0 parameters ✓
# Learned: max_seq_len × d_model parameters ✗
# RoPE: 0 parameters ✓ (just cos/sin cache)
```

Combines benefits of sinusoidal (no parameters) with benefits of relative encoding!

---

## 🚀 RoPE in Modern LLMs

### LLaMA Architecture
```python
class LLaMAAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionEmbedding(d_model // num_heads)  # ← RoPE!

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Apply RoPE to Q and K (NOT V!)
        Q, K = self.rope(Q, K)

        # Standard attention
        attention = scaled_dot_product_attention(Q, K, V)
        return attention
```

**Notice**: RoPE applied to Q and K, but NOT V!
- Q and K determine attention scores (need position info)
- V contains content (position already captured by Q/K)

### Mistral 7B Configuration
```python
# Mistral uses RoPE with specific parameters
d_model = 4096
num_heads = 32
head_dim = 128
rope_base = 10000  # Standard base
max_position = 32768  # Can handle very long contexts!

rope = RotaryPositionEmbedding(dim=head_dim, base=rope_base)
```

### GPT-Neo Configuration
```python
# GPT-Neo (EleutherAI) also uses RoPE
d_model = 2048
num_heads = 16
head_dim = 128

rope = RotaryPositionEmbedding(dim=head_dim)
```

---

## 🔬 Advanced: RoPE Variants

### 1. Linear RoPE (LongRoPE)

For extreme long-context extension:

```python
# Scale rotation frequencies for longer contexts
rope = RotaryPositionEmbedding(dim=64, base=10000 * scale_factor)
```

Used to extend LLaMA from 4k → 100k+ tokens!

### 2. Dynamic RoPE

Adjust base during inference for different context lengths:

```python
def dynamic_rope_base(seq_len, original_max_len, base=10000):
    if seq_len <= original_max_len:
        return base
    scale = seq_len / original_max_len
    return base * scale
```

### 3. Yarn (Yet Another RoPE Variant)

Sophisticated frequency interpolation for context extension:
- Used in some extended-context LLaMA variants
- Preserves both local and global patterns

---

## 💡 Key Implementation Details

### Why Rotate Half?

```python
def rotate_half(x):
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    return torch.cat([-x2, x1], dim=-1)
```

This implements the rotation formula efficiently:
- Instead of explicit matrix multiplication
- Leverages the structure of rotation matrices
- Much faster on GPUs!

### Caching for Efficiency

```python
# Pre-compute and cache cos/sin for all positions
self.register_buffer('cos_cached', ...)
self.register_buffer('sin_cached', ...)
```

**Why cache?**
- Cos/sin values are the same for every forward pass
- Compute once, reuse forever
- Significant speedup during training and inference

### Applying Only to Q and K

```python
Q, K = rope(Q, K)  # Rotate queries and keys
# V not rotated!
```

**Why?**
- Position info needed to compute attention scores (Q @ K^T)
- Values contain content, which is position-independent
- Saves computation!

---

## 📊 RoPE vs Traditional Position Encoding

| Feature | Sinusoidal | Learned | RoPE |
|---------|-----------|---------|------|
| **Parameters** | 0 | O(L × d) | 0 |
| **Extrapolation** | Good | Poor | Excellent |
| **Relative encoding** | Implicit | No | Explicit |
| **Integration** | Additive | Additive | Multiplicative |
| **Used in modern LLMs** | Rare | Some | Most |

**Winner**: RoPE dominates modern architectures!

---

## 🎓 Key Takeaways

1. **Rotation instead of addition**: RoPE rotates Q/K vectors by position-dependent angles
2. **Relative position encoding**: Attention scores depend on relative distances, not absolute positions
3. **Perfect extrapolation**: Seamlessly handles sequences longer than training length
4. **Zero parameters**: No learned embeddings, just cached cos/sin values
5. **Industry standard**: Used in LLaMA, PaLM, Mistral, GPT-Neo, and most modern LLMs
6. **Multi-scale**: Different frequencies capture different-range dependencies
7. **Efficient**: Minimal computational overhead, easily cached

**RoPE is THE position encoding method for modern LLMs!**

---

## 📝 Quiz Time!

Test your understanding of RoPE.
""")

# Quiz questions
with st.expander("Question 1: Why is RoPE better than additive position encoding?"):
    st.markdown("""
    **Question**: What is the PRIMARY advantage of RoPE over traditional additive positional encoding?

    A) RoPE uses fewer parameters
    B) RoPE encodes relative positions directly in attention scores
    C) RoPE is faster to compute
    D) RoPE works with smaller models
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) RoPE encodes relative positions directly in attention scores**

        **Explanation**:

        **Traditional (additive):**
        ```python
        x = embedding + positional_encoding
        Q, K = project(x)
        scores = Q @ K^T  # Position info indirect, diluted
        ```

        **RoPE (rotary):**
        ```python
        Q, K = project(embedding)
        Q, K = rotate(Q, K, position)
        scores = Q @ K^T  # Position directly in attention scores!
        ```

        **Key property**: `score(m, n) = f(m - n)` - depends ONLY on relative distance!

        This means:
        - Model learns "attend to word 3 positions back"
        - Not "attend to absolute position 5"
        - Generalizes better, especially to longer sequences
        """)

with st.expander("Question 2: Why apply RoPE to Q and K but not V?"):
    st.markdown("""
    **Question**: Why do we apply RoPE to queries (Q) and keys (K) but NOT values (V)?

    A) Values are too large to rotate
    B) Position information is needed for attention scores (Q@K^T), not for content (V)
    C) It's faster to only rotate two matrices
    D) Values already contain position information
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) Position information is needed for attention scores (Q@K^T), not for content (V)**

        **Explanation**:

        Attention mechanism:
        ```python
        scores = Q @ K^T  # ← Position needed here!
        attention_weights = softmax(scores)
        output = attention_weights @ V  # ← Just content retrieval
        ```

        **Q and K**: Determine WHICH positions to attend to
        - Need position information to compute relevance
        - "Should I attend to the word 3 positions back?"

        **V**: Contains the actual CONTENT to retrieve
        - Position-independent information
        - "What information does this word contain?"

        **Analogy**: In a library:
        - Q (query) and K (book titles): Need to know shelf positions
        - V (book content): Content is the same regardless of shelf position
        """)

with st.expander("Question 3: RoPE extrapolation"):
    st.markdown("""
    **Question**: Why can RoPE extrapolate to longer sequences better than learned positional embeddings?

    A) RoPE uses more parameters
    B) RoPE is a continuous mathematical function with no fixed length limit
    C) RoPE trains faster
    D) RoPE uses less memory
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) RoPE is a continuous mathematical function with no fixed length limit**

        **Explanation**:

        **Learned embeddings:**
        ```python
        position_emb = nn.Embedding(max_position=2048, dim=512)
        # Trained on positions 0-2047
        position_emb[5000]  # ERROR! Out of bounds!
        ```

        **RoPE:**
        ```python
        # Rotation by angle = position × frequency
        angle = position × θ
        # Mathematical function works for ANY position!
        rope.rotate(position=5000)  # Works fine! ✓
        ```

        **Real-world impact:**
        - LLaMA trained on 4k tokens → works on 32k+ with RoPE
        - Mistral trained on 8k → extends to 32k
        - Critical for long-document understanding

        RoPE treats position as a continuous variable, not discrete indices!
        """)

with st.expander("Question 4: RoPE rotation frequencies"):
    st.markdown("""
    **Question**: In RoPE, why do different dimension pairs use different rotation frequencies?

    A) To reduce computational cost
    B) To capture both short-range and long-range position dependencies at multiple scales
    C) To make the model deeper
    D) To reduce memory usage
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) To capture both short-range and long-range position dependencies at multiple scales**

        **Explanation**:

        ```python
        # Different frequencies for different dimension pairs
        θ₀ = 1.0      # Slow rotation → long-range patterns
        θ₁ = 0.1      # Medium rotation → medium-range patterns
        θ₂ = 0.01     # Fast rotation → short-range patterns
        θ₃ = 0.001    # Very fast rotation → very short-range patterns
        ```

        **Multi-scale representation:**
        - **Low frequencies**: Distinguish "beginning vs. end" of document
        - **Medium frequencies**: Distinguish "nearby vs. distant" paragraphs
        - **High frequencies**: Distinguish "adjacent vs. separated" words

        **Analogy**: Like a clock with multiple hands
        - Hour hand: Long-range (slow rotation)
        - Minute hand: Medium-range
        - Second hand: Short-range (fast rotation)

        Each frequency band captures patterns at a different scale!
        """)

with st.expander("Question 5: RoPE adoption in modern LLMs"):
    st.markdown("""
    **Question**: Which of the following modern LLMs use RoPE?

    A) Only LLaMA
    B) Only open-source models
    C) LLaMA, PaLM, Mistral, GPT-Neo, and most LLMs since 2021
    D) No production models use RoPE
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: C) LLaMA, PaLM, Mistral, GPT-Neo, and most LLMs since 2021**

        **Explanation**:

        **Models using RoPE:**

        **Open Source:**
        - LLaMA 1, 2, 3 (Meta)
        - Mistral, Mixtral (Mistral AI)
        - GPT-Neo, GPT-J (EleutherAI)
        - Falcon (TII)
        - Qwen, Yi (Chinese models)

        **Closed Source:**
        - PaLM, Gemini (Google)
        - Many others (proprietary)

        **Why universal adoption?**
        - Better long-context performance
        - Perfect extrapolation
        - Zero extra parameters
        - Relative position encoding

        **The shift:**
        - 2017-2020: Sinusoidal or learned embeddings
        - 2021+: RoPE becomes the standard

        RoPE is now the DEFAULT choice for new transformer architectures!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand RoPE - the position encoding that powers modern LLMs!

With attention, multi-head attention, and position encoding under your belt, you're ready for the complete picture:

Next topics:
- **Topic 18**: The Transformer Architecture - putting ALL the pieces together (attention + position + feedforward + normalization)
- **Topic 19**: Modern Transformer Components - how LLaMA/GPT differ from the original (RMSNorm, SwiGLU, etc.)
- **Topic 20**: Grouped Query Attention - memory-efficient attention for large models

**You now understand the position encoding used in LLaMA, Mistral, PaLM, and most modern LLMs!** 🚀

---

*Navigation: ← Previous: Topic 16 (Positional Encoding) | Next: Topic 18 (Transformer Architecture) →*
""")
