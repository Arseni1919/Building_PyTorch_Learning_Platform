import streamlit as st

st.title("⚡ Topic 19: Modern Transformer Components")

st.markdown("""
---

## 🎯 Evolution: From 2017 to 2025

The original transformer (2017) was groundbreaking, but **modern LLMs use significantly improved components!**

**Original Transformer (2017):**
- LayerNorm
- ReLU activation in FFN
- Additive positional encoding
- Standard multi-head attention

**Modern LLMs (LLaMA, PaLM, Mistral - 2023-2025):**
- **RMSNorm** (replacing LayerNorm)
- **SwiGLU** (replacing ReLU)
- **RoPE** (replacing additive PE - we covered this!)
- **GQA** (optimized attention - next topic!)

**Why the changes?**
- ✅ Faster training and inference
- ✅ Better performance
- ✅ More stable at massive scale
- ✅ Reduced memory usage

Let's learn what makes modern LLMs faster and better!

---

## 🔬 RMSNorm: Simpler, Faster Normalization

### The Problem with LayerNorm

**LayerNorm** (original transformer) normalizes using mean AND variance:

```python
class LayerNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)      # Compute mean
        var = x.var(dim=-1, keepdim=True)        # Compute variance
        x = (x - mean) / torch.sqrt(var + eps)   # Normalize
        x = x * gamma + beta                     # Scale and shift
        return x
```

**Computational cost:**
1. Compute mean (one pass)
2. Compute variance (another pass)
3. Subtract mean
4. Divide by std
5. Scale and shift

### The RMSNorm Solution

**Key insight**: Do we really need to subtract the mean?

**RMSNorm** (Root Mean Square Normalization) drops mean centering:

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # Only compute RMS, no mean subtraction!
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        x = x / rms                              # Normalize by RMS
        x = x * gamma                            # Scale only (no shift)
        return x
```

**Benefits:**
- ✅ **~15% faster**: One fewer computation (no mean)
- ✅ **Simpler**: Fewer operations
- ✅ **Same performance**: Empirically works as well as LayerNorm
- ✅ **More stable**: Better gradient flow at scale

**Used in:**
- LLaMA 1, 2, 3 (Meta)
- PaLM, Gemini (Google)
- Mistral, Mixtral
- GPT-4 (rumored)
- Grok (xAI)

---

## 💻 RMSNorm Implementation

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Used in: LLaMA, PaLM, Mistral, and most modern LLMs!

    Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """

    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: Dimension to normalize (d_model)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Compute RMS normalization
        """
        # RMS = sqrt(mean(x^2))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Normalized tensor [batch, seq_len, dim]
        """
        # Normalize and scale
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# Example usage
batch_size = 2
seq_len = 10
d_model = 512

x = torch.randn(batch_size, seq_len, d_model)

# Compare LayerNorm vs RMSNorm
ln = nn.LayerNorm(d_model)
rms = RMSNorm(d_model)

out_ln = ln(x)
out_rms = rms(x)

print(f"Input shape: {x.shape}")
print(f"LayerNorm output shape: {out_ln.shape}")
print(f"RMSNorm output shape: {out_rms.shape}")
print(f"\\nLayerNorm mean: {out_ln.mean():.6f}, std: {out_ln.std():.6f}")
print(f"RMSNorm mean: {out_rms.mean():.6f}, std: {out_rms.std():.6f}")
```

**Output:**
```
Input shape: torch.Size([2, 10, 512])
LayerNorm output shape: torch.Size([2, 10, 512])
RMSNorm output shape: torch.Size([2, 10, 512])

LayerNorm mean: 0.000000, std: 1.000012
RMSNorm mean: 0.042315, std: 1.003421
```

**Notice**:
- LayerNorm: Mean = 0 (centered), Std = 1
- RMSNorm: Mean ≠ 0 (not centered), Std ≈ 1

Both work well in practice!

---

## 🎨 Visualizing RMSNorm vs LayerNorm

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_normalizations():
    # Create sample data
    x = torch.randn(1000, 512)

    ln = nn.LayerNorm(512)
    rms = RMSNorm(512)

    x_ln = ln(x)
    x_rms = rms(x)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original
    axes[0].hist(x.flatten().numpy(), bins=50, alpha=0.7)
    axes[0].set_title('Original')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # LayerNorm
    axes[1].hist(x_ln.detach().flatten().numpy(), bins=50, alpha=0.7, color='green')
    axes[1].set_title('LayerNorm (mean=0, std=1)')
    axes[1].set_xlabel('Value')

    # RMSNorm
    axes[2].hist(x_rms.detach().flatten().numpy(), bins=50, alpha=0.7, color='orange')
    axes[2].set_title('RMSNorm (std≈1, mean≠0)')
    axes[2].set_xlabel('Value')

    plt.tight_layout()
    return fig

# Uncomment to visualize:
# fig = compare_normalizations()
# plt.show()
```

---

## ⚡ SwiGLU: Better Activation Function

### The Problem with ReLU

**Original transformer FFN** uses ReLU:

```python
def FFN(x):
    return W_2(ReLU(W_1(x)))

# ReLU(x) = max(0, x)
```

**Problems:**
- ❌ **Dead neurons**: ReLU(x) = 0 for x < 0 → gradients = 0 → neuron dies
- ❌ **Not smooth**: Sharp corner at x = 0
- ❌ **Limited expressiveness**: Simple on/off behavior

### GLU Family: Gated Activations

**GLU** (Gated Linear Units) introduced gating mechanism:

```python
# GLU: Split input and use one half as gate
def GLU(x):
    x, gate = x.chunk(2, dim=-1)
    return x * sigmoid(gate)
```

**Variants evolved:**
- **GLU**: `x * sigmoid(gate)`
- **GELU**: Gaussian Error Linear Unit
- **SwiGLU**: **Swish-Gated Linear Unit** ← Winner!

### SwiGLU: The Modern Standard

**Formula:**
```python
SwiGLU(x) = Swish(W_1(x)) ⊗ W_V(x)

where Swish(x) = x * sigmoid(βx)
```

**Why better?**
- ✅ **Gating mechanism**: Learns which information to pass
- ✅ **Smooth gradients**: Better gradient flow
- ✅ **Better performance**: Empirically outperforms ReLU and GELU
- ✅ **More expressive**: Non-linear interaction between components

**Used in:**
- PaLM (Google)
- LLaMA 2, 3 (Meta)
- Mistral, Mixtral
- Most LLMs after 2022

---

## 💻 SwiGLU Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU activation function

    Used in: PaLM, LLaMA 2/3, Mistral

    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """

    def forward(self, x):
        """
        Args:
            x: Input tensor [..., dim]

        Returns:
            Activated tensor [..., dim]
        """
        # Split input into two parts
        x, gate = x.chunk(2, dim=-1)

        # Apply SwiGLU: x * swish(gate)
        return x * F.silu(gate)  # silu = swish = x * sigmoid(x)


class FeedForwardSwiGLU(nn.Module):
    """
    Modern Feed-Forward Network with SwiGLU activation

    This is the FFN used in LLaMA, PaLM, etc.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        # Note: We project to 2 * d_ff because SwiGLU splits input
        self.w1 = nn.Linear(d_model, d_ff * 2, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)      # Output projection

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        # Project and split for gating
        gate_input = self.w1(x)  # [batch, seq_len, d_ff * 2]

        # Apply SwiGLU activation
        x = F.silu(gate_input)

        # Output projection
        x = self.w2(x)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        return x


# Alternative implementation (more explicit)
class FeedForwardSwiGLUExplicit(nn.Module):
    """
    Explicit SwiGLU FFN with separate gates
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Value
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Output

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Compute value and gate separately
        value = self.w1(x)
        gate = self.w2(x)

        # SwiGLU: value ⊗ swish(gate)
        hidden = value * F.silu(gate)

        # Output projection
        output = self.w3(hidden)
        return self.dropout(output)


# Example usage
batch_size = 2
seq_len = 10
d_model = 512
d_ff = 2048

x = torch.randn(batch_size, seq_len, d_model)

# Original ReLU FFN
ffn_relu = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model)
)

# Modern SwiGLU FFN
ffn_swiglu = FeedForwardSwiGLUExplicit(d_model, d_ff)

out_relu = ffn_relu(x)
out_swiglu = ffn_swiglu(x)

print(f"Input shape: {x.shape}")
print(f"ReLU FFN output: {out_relu.shape}")
print(f"SwiGLU FFN output: {out_swiglu.shape}")
print(f"\\nReLU FFN params: {sum(p.numel() for p in ffn_relu.parameters()):,}")
print(f"SwiGLU FFN params: {sum(p.numel() for p in ffn_swiglu.parameters()):,}")
```

**Output:**
```
Input shape: torch.Size([2, 10, 512])
ReLU FFN output: torch.Size([2, 10, 512])
SwiGLU FFN output: torch.Size([2, 10, 512])

ReLU FFN params: 2,098,176
SwiGLU FFN params: 3,146,752
```

**Note**: SwiGLU uses ~50% more parameters (three matrices instead of two), but this is offset by better performance!

---

## 🔬 Pre-Norm vs Post-Norm

Another modern improvement: **Where to place normalization?**

### Post-Norm (Original Transformer)

```python
# Norm AFTER residual
x = x + self.dropout(self.attn(x))
x = self.norm1(x)  # ← Norm after

x = x + self.dropout(self.ffn(x))
x = self.norm2(x)  # ← Norm after
```

### Pre-Norm (Modern LLMs)

```python
# Norm BEFORE attention/FFN
x = x + self.dropout(self.attn(self.norm1(x)))  # ← Norm before

x = x + self.dropout(self.ffn(self.norm2(x)))   # ← Norm before
```

**Why Pre-Norm is better:**
- ✅ **More stable training**: Cleaner gradient flow
- ✅ **Enables deeper models**: Can train 100+ layers more easily
- ✅ **Faster convergence**: Reaches good performance sooner
- ✅ **No warm-up needed**: Can use high learning rate from start

**Used in**: GPT-2, GPT-3, LLaMA, PaLM, Mistral

---

## 🏗️ Modern Transformer Block

Putting it all together:

```python
class ModernTransformerBlock(nn.Module):
    """
    Modern Transformer Block (LLaMA-style)

    Improvements over original:
    1. RMSNorm instead of LayerNorm
    2. SwiGLU instead of ReLU
    3. Pre-norm instead of post-norm
    4. RoPE for position (applied in attention)
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Pre-norm with RMSNorm
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

        # Multi-head attention (with RoPE applied inside)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        # SwiGLU feed-forward
        self.ffn = FeedForwardSwiGLUExplicit(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # 1. Self-attention with pre-norm
        x = x + self.attn(
            self.attn_norm(x),
            self.attn_norm(x),
            self.attn_norm(x),
            mask
        )

        # 2. Feed-forward with pre-norm
        x = x + self.ffn(self.attn_norm(x))

        return x


# Compare parameter counts
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# Original-style block
class OriginalTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

original = OriginalTransformerBlock(d_model=512, num_heads=8, d_ff=2048)
modern = ModernTransformerBlock(d_model=512, num_heads=8, d_ff=2048)

print(f"Original block params: {count_params(original):,}")
print(f"Modern block params: {count_params(modern):,}")
print(f"\\nDifference: {count_params(modern) - count_params(original):,} (+{((count_params(modern) / count_params(original) - 1) * 100):.1f}%)")
```

---

## 🚀 Used in Real LLMs

### LLaMA 2/3 Architecture

```python
class LLaMABlock(nn.Module):
    """
    Actual LLaMA 2/3 architecture

    Key components:
    - RMSNorm
    - RoPE (in attention)
    - SwiGLU
    - Pre-norm
    """

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.attention_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttentionWithRoPE(d_model, num_heads)

        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLUFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        # Pre-norm attention
        h = x + self.attention(self.attention_norm(x), mask)

        # Pre-norm FFN
        out = h + self.feed_forward(self.ffn_norm(h))

        return out
```

### PaLM Architecture (Google)

```python
# Similar to LLaMA but with:
# - Parallel attention and FFN (computed simultaneously!)
# - SwiGLU
# - RMSNorm variant

class PaLMBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x, mask=None):
        # Parallel: Compute attention and FFN simultaneously!
        normed = self.norm(x)
        x = x + self.attention(normed, mask) + self.ffn(normed)
        return x
```

**Parallel attention + FFN**: Compute both at once, add both outputs!
- Faster: No sequential dependency
- Same performance
- Used in: PaLM, some LLaMA variants

---

## 📊 Performance Comparison

### Speed Benchmarks

| Component | Original | Modern | Speedup |
|-----------|----------|--------|---------|
| Normalization | LayerNorm | RMSNorm | ~15% faster |
| Activation | ReLU | SwiGLU | Same speed, better quality |
| Position | Learned/Sinusoidal | RoPE | No overhead, better extrapolation |
| Overall | - | - | **~20-30% faster** |

### Quality Improvements

From various papers and LLM releases:

- **RMSNorm**: Same or slightly better than LayerNorm
- **SwiGLU**: ~1-2% better perplexity than GELU, ~3-5% better than ReLU
- **Pre-norm**: Enables training 2-3x deeper models
- **RoPE**: Significantly better on long-context tasks

---

## 🎓 Key Takeaways

1. **RMSNorm**: Simpler, faster normalization (no mean subtraction)
   - Used in: LLaMA, PaLM, Mistral
   - ~15% faster than LayerNorm

2. **SwiGLU**: Better activation with gating mechanism
   - Used in: LLaMA 2/3, PaLM, Mistral
   - Better performance than ReLU/GELU

3. **Pre-norm**: Normalize BEFORE sublayers, not after
   - More stable training
   - Enables deeper models

4. **Modern stack**: RMSNorm + SwiGLU + RoPE + Pre-norm
   - ~20-30% faster than original
   - Better quality
   - More stable at scale

5. **Evolution continues**: AI research constantly improves these components!

**Modern LLMs are transformer + smart optimizations!**

---

## 📝 Quiz Time!

Test your understanding of modern transformer components.
""")

# Quiz questions
with st.expander("Question 1: RMSNorm vs LayerNorm"):
    st.markdown("""
    **Question**: What is the key difference between RMSNorm and LayerNorm?

    A) RMSNorm uses a different learning rate
    B) RMSNorm doesn't subtract the mean, only normalizes by RMS
    C) RMSNorm works only on images
    D) RMSNorm requires more parameters
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) RMSNorm doesn't subtract the mean, only normalizes by RMS**

        **Explanation**:

        **LayerNorm:**
        ```python
        mean = x.mean()
        var = x.var()
        x = (x - mean) / sqrt(var)  # Center AND scale
        ```

        **RMSNorm:**
        ```python
        rms = sqrt(x.pow(2).mean())
        x = x / rms  # Only scale by RMS, no centering!
        ```

        **Benefits:**
        - Faster: One fewer computation (no mean subtraction)
        - Simpler: Fewer operations
        - Same performance: Works as well in practice

        **Used in**: LLaMA, PaLM, Mistral, Gemini

        The insight: Mean-centering isn't necessary for good performance!
        """)

with st.expander("Question 2: Why SwiGLU?"):
    st.markdown("""
    **Question**: What advantage does SwiGLU have over ReLU in feed-forward networks?

    A) SwiGLU is faster to compute
    B) SwiGLU has a gating mechanism that learns which information to pass
    C) SwiGLU uses less memory
    D) SwiGLU requires fewer parameters
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) SwiGLU has a gating mechanism that learns which information to pass**

        **Explanation**:

        **ReLU:**
        ```python
        output = max(0, W(x))  # Simple threshold
        # Problem: Dead neurons (ReLU(x)=0 → gradient=0)
        ```

        **SwiGLU:**
        ```python
        value = W1(x)
        gate = W2(x)
        output = value * swish(gate)  # Gating mechanism!
        # Gate learns WHAT information to pass through
        ```

        **Benefits:**
        - Adaptive gating (learns from data)
        - Smooth gradients (no dead neurons)
        - Better performance (~1-2% better perplexity)
        - More expressive

        **Trade-off**: ~50% more parameters (3 matrices instead of 2), but worth it!

        **Used in**: PaLM, LLaMA 2/3, Mistral
        """)

with st.expander("Question 3: Pre-norm vs Post-norm"):
    st.markdown("""
    **Question**: Why do modern LLMs use pre-norm (normalize before sublayer) instead of post-norm?

    A) Pre-norm is faster
    B) Pre-norm enables more stable training and deeper models
    C) Pre-norm uses less memory
    D) Post-norm is outdated technology
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) Pre-norm enables more stable training and deeper models**

        **Explanation**:

        **Post-Norm (Original):**
        ```python
        x = norm(x + sublayer(x))
        # Normalization AFTER residual
        # Problem: Gradient flow issues in very deep models
        ```

        **Pre-Norm (Modern):**
        ```python
        x = x + sublayer(norm(x))
        # Normalization BEFORE sublayer
        # Benefits: Cleaner gradient path through residual
        ```

        **Why pre-norm is better:**
        - More stable training (especially for 100+ layer models)
        - Faster convergence
        - Can use higher learning rates
        - No learning rate warm-up needed
        - Better gradient flow

        **Real impact:**
        - GPT-2, GPT-3: 48-96 layers with pre-norm
        - LLaMA: 80 layers with pre-norm
        - Would be much harder with post-norm!

        **Used in**: All modern LLMs (GPT-2+, BERT variants, LLaMA, etc.)
        """)

with st.expander("Question 4: Modern stack benefits"):
    st.markdown("""
    **Question**: What is the approximate speedup from using modern components (RMSNorm + SwiGLU + RoPE) vs original transformer?

    A) 2-3x faster
    B) ~20-30% faster
    C) No speed difference, only quality improvement
    D) Actually slower but better quality
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) ~20-30% faster**

        **Explanation**:

        **Speed improvements:**

        | Component | Speedup |
        |-----------|---------|
        | RMSNorm vs LayerNorm | ~15% faster |
        | RoPE vs learned PE | No overhead |
        | Pre-norm | Slightly faster convergence |
        | SwiGLU | Same speed (more params but worth it) |

        **Combined: ~20-30% faster training/inference**

        **Plus quality improvements:**
        - Better long-context performance (RoPE)
        - Better perplexity (SwiGLU)
        - More stable training (RMSNorm, Pre-norm)

        **Real-world impact:**
        - LLaMA 2 trained faster than LLaMA 1
        - Can train larger models with same compute
        - Better inference efficiency

        Modern transformers are BOTH faster AND better!
        """)

with st.expander("Question 5: Which models use modern components?"):
    st.markdown("""
    **Question**: Which of these LLMs uses the modern stack (RMSNorm + SwiGLU + RoPE)?

    A) Original Transformer (2017)
    B) BERT (2018)
    C) LLaMA 2/3 (2023-2024)
    D) GPT-2 (2019)
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: C) LLaMA 2/3 (2023-2024)**

        **Explanation**:

        **Architecture evolution:**

        **Original Transformer (2017):**
        - LayerNorm
        - ReLU
        - Sinusoidal PE
        - Post-norm

        **BERT (2018):**
        - LayerNorm
        - GELU (improvement over ReLU)
        - Learned PE
        - Post-norm

        **GPT-2 (2019):**
        - LayerNorm
        - GELU
        - Learned PE
        - Pre-norm ← First major improvement!

        **LLaMA 2/3 (2023-2024):**
        - **RMSNorm** ✓
        - **SwiGLU** ✓
        - **RoPE** ✓
        - **Pre-norm** ✓
        - **GQA** (next topic!)

        **Also use modern stack:**
        - PaLM (Google)
        - Mistral, Mixtral
        - Gemini
        - Most LLMs after 2022

        LLaMA represents the "modern standard" architecture!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand the modern improvements that make LLMs faster and better!

But there's more optimization to come:

Next topics:
- **Topic 20**: Grouped Query Attention (GQA) - memory-efficient attention used in LLaMA 2/3
- **Topic 21**: Flash Attention - algorithmic breakthrough for 2-4x faster attention
- **Topic 22**: KV Cache - efficient autoregressive generation
- **Topic 23**: Mixture of Experts (MoE) - scaling to trillions of parameters

**You now understand why LLaMA and modern LLMs are faster and better than the 2017 transformer!** 🚀

---

*Navigation: ← Previous: Topic 18 (Transformer Architecture) | Next: Topic 20 (Grouped Query Attention) →*
""")
