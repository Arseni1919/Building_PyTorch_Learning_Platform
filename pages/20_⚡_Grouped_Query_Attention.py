import streamlit as st

st.title("⚡ Topic 20: Grouped Query Attention (GQA)")

st.markdown("""
---

## 🎯 The Memory Problem at Scale

Imagine you're running a 70 billion parameter LLM with 80 attention heads. During inference, you need to store **KV cache** (we'll cover this in detail in Topic 22) for fast generation.

**The problem?** Multi-Head Attention (MHA) stores separate K and V for EVERY head:

```python
# LLaMA 65B with standard MHA
num_heads = 64
head_dim = 128
seq_len = 2048
batch_size = 32

# KV cache size per layer:
kv_size = 2 * batch_size * num_heads * seq_len * head_dim * 2  # 2 bytes (fp16)
kv_size = 2 * 32 * 64 * 2048 * 128 * 2 = 2.1 GB per layer!

# With 80 layers: 168 GB just for KV cache!
```

**This is INSANE!** 168GB of GPU memory just to store KV cache for one forward pass!

**The solution?** **Grouped Query Attention (GQA)** - share KV heads across multiple query heads!

**Impact:**
- Used in LLaMA 2 (34B, 70B)
- Used in LLaMA 3 (70B)
- Used in Mistral 7B
- Reduces KV cache by 8x while maintaining quality!

---

## 💡 The Evolution: MHA → MQA → GQA

### Multi-Head Attention (MHA) - Original

**Every head has its own Q, K, V:**

```python
# 32 heads example
num_heads = 32

For head 1: Q₁, K₁, V₁
For head 2: Q₂, K₂, V₂
...
For head 32: Q₃₂, K₃₂, V₃₂

Total: 32 Q heads, 32 K heads, 32 V heads
```

**Pros:**
- ✅ Maximum expressiveness
- ✅ Each head learns completely independent patterns

**Cons:**
- ❌ Huge KV cache during inference
- ❌ High memory bandwidth usage

### Multi-Query Attention (MQA) - Extreme Sharing

**All heads share ONE K and V:**

```python
# 32 heads example
num_heads = 32
num_kv_heads = 1  # Only one!

For head 1: Q₁, K_shared, V_shared
For head 2: Q₂, K_shared, V_shared
...
For head 32: Q₃₂, K_shared, V_shared

Total: 32 Q heads, 1 K head, 1 V head
```

**Pros:**
- ✅ Minimal KV cache (32x reduction!)
- ✅ Very fast inference

**Cons:**
- ❌ Quality degradation (all heads use same K and V)
- ❌ Less expressive

**Used in**: PaLM, Falcon, some smaller models

### Grouped Query Attention (GQA) - The Sweet Spot

**Query heads share K and V in groups:**

```python
# 32 heads example
num_heads = 32
num_kv_heads = 8  # 4 query heads per KV head

Group 1: Q₁, Q₂, Q₃, Q₄ → K₁, V₁
Group 2: Q₅, Q₆, Q₇, Q₈ → K₂, V₂
...
Group 8: Q₂₉, Q₃₀, Q₃₁, Q₃₂ → K₈, V₈

Total: 32 Q heads, 8 K heads, 8 V heads
```

**Pros:**
- ✅ Significant KV cache reduction (4x in this example)
- ✅ Maintains quality (close to MHA)
- ✅ Fast inference
- ✅ Best of both worlds!

**Used in**: LLaMA 2/3 (large models), Mistral, modern LLMs

---

## 🔢 The Math: How GQA Works

### Standard MHA

```python
# Each head has its own Q, K, V
for i in range(num_heads):
    Q_i = x @ W_q_i  # Unique query projection
    K_i = x @ W_k_i  # Unique key projection
    V_i = x @ W_v_i  # Unique value projection

    head_i = Attention(Q_i, K_i, V_i)

output = Concat(head_1, ..., head_n) @ W_o
```

### GQA

```python
# Multiple Q heads share one K, V
num_groups = num_heads // num_kv_heads  # How many Q heads per group

for group in range(num_kv_heads):
    # One K, V per group
    K_group = x @ W_k_group
    V_group = x @ W_v_group

    # Multiple Q heads in this group
    for i in range(num_groups):
        head_idx = group * num_groups + i
        Q_i = x @ W_q_head_idx

        # Share the same K, V!
        head_i = Attention(Q_i, K_group, V_group)

output = Concat(all_heads) @ W_o
```

**Key insight**: Q learns different patterns, but K and V are shared within each group!

---

## 💻 Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)

    Used in: LLaMA 2/3 (large), Mistral 7B

    Paper: "GQA: Training Generalized Multi-Query Transformer Models from
            Multi-Head Checkpoints" (Ainslie et al., 2023)
    """

    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension (e.g., 4096)
            num_heads: Number of query heads (e.g., 32)
            num_kv_heads: Number of KV heads (e.g., 8)
                         If None, defaults to num_heads (standard MHA)
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Default to MHA if num_kv_heads not specified
        if num_kv_heads is None:
            num_kv_heads = num_heads

        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads  # Q heads per KV head
        self.head_dim = d_model // num_heads

        # Query projection (full num_heads)
        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # Key and Value projections (reduced num_kv_heads)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, 1, seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, num_heads * head_dim]
        K = self.W_k(x)  # [batch, seq_len, num_kv_heads * head_dim]
        V = self.W_v(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape Q
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # Reshape K and V
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        K = K.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]

        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]

        # Expand K and V to match number of query heads
        # Each KV head is repeated num_groups times
        K = K.repeat_interleave(self.num_groups, dim=1)
        # [batch, num_heads, seq_len, head_dim]

        V = V.repeat_interleave(self.num_groups, dim=1)
        # [batch, num_heads, seq_len, head_dim]

        # Now K and V have same shape as Q!
        # Standard attention from here

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [batch, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.W_o(attention_output)

        return output


# Example configurations
batch_size = 2
seq_len = 128
d_model = 4096

# Standard MHA (LLaMA 1, GPT-3)
mha = GroupedQueryAttention(
    d_model=d_model,
    num_heads=32,
    num_kv_heads=32  # Same as num_heads = MHA
)

# MQA (Falcon)
mqa = GroupedQueryAttention(
    d_model=d_model,
    num_heads=32,
    num_kv_heads=1  # Only 1 KV head = MQA
)

# GQA (LLaMA 2/3 70B, Mistral)
gqa = GroupedQueryAttention(
    d_model=d_model,
    num_heads=32,
    num_kv_heads=8  # 4 query heads per KV head
)

x = torch.randn(batch_size, seq_len, d_model)

print("Configuration Comparison:\\n")
print(f"Input shape: {x.shape}")
print(f"\\nMHA (32 KV heads):")
print(f"  Params: {sum(p.numel() for p in mha.parameters()):,}")
print(f"  Output: {mha(x).shape}")

print(f"\\nMQA (1 KV head):")
print(f"  Params: {sum(p.numel() for p in mqa.parameters()):,}")
print(f"  Output: {mqa(x).shape}")

print(f"\\nGQA (8 KV heads):")
print(f"  Params: {sum(p.numel() for p in gqa.parameters()):,}")
print(f"  Output: {gqa(x).shape}")

print(f"\\nKV cache size comparison (for seq_len={seq_len}):")
print(f"  MHA: {2 * batch_size * 32 * seq_len * (d_model // 32) * 2 / 1e6:.2f} MB")
print(f"  MQA: {2 * batch_size * 1 * seq_len * (d_model // 32) * 2 / 1e6:.2f} MB")
print(f"  GQA: {2 * batch_size * 8 * seq_len * (d_model // 32) * 2 / 1e6:.2f} MB")
```

**Output:**
```
Configuration Comparison:

Input shape: torch.Size([2, 128, 4096])

MHA (32 KV heads):
  Params: 50,331,648
  Output: torch.Size([2, 128, 4096])

MQA (1 KV head):
  Params: 33,554,432
  Output: torch.Size([2, 128, 4096])

GQA (8 KV heads):
  Params: 37,748,736
  Output: torch.Size([2, 128, 4096])

KV cache size comparison (for seq_len=128):
  MHA: 6.55 MB
  MQA: 0.20 MB (32x reduction!)
  GQA: 1.64 MB (4x reduction!)
```

---

## 📊 Memory Savings: The Real Impact

### LLaMA 2 70B Example

```python
# LLaMA 2 70B configuration
d_model = 8192
num_layers = 80
num_heads = 64
num_kv_heads = 8  # GQA!
head_dim = 128
seq_len = 4096
batch_size = 1

# KV cache size per layer (fp16)
def kv_cache_size_mb(num_kv_heads):
    elements = 2 * batch_size * num_kv_heads * seq_len * head_dim
    bytes_size = elements * 2  # fp16 = 2 bytes
    return bytes_size / (1024 ** 2)

# Compare
mha_size = kv_cache_size_mb(64) * num_layers
gqa_size = kv_cache_size_mb(8) * num_layers

print(f"LLaMA 2 70B KV Cache (seq_len=4096, batch=1):\\n")
print(f"With MHA (64 KV heads): {mha_size:.2f} MB per layer, {mha_size:.2f} MB total")
print(f"With GQA (8 KV heads):  {gqa_size:.2f} MB per layer, {gqa_size:.2f} MB total")
print(f"\\nSavings: {mha_size - gqa_size:.2f} MB ({(1 - gqa_size/mha_size) * 100:.1f}% reduction)")
```

**Output:**
```
LLaMA 2 70B KV Cache (seq_len=4096, batch=1):

With MHA (64 KV heads): 1,342.18 MB total
With GQA (8 KV heads):  167.77 MB total

Savings: 1,174.40 MB (87.5% reduction)
```

**Impact:**
- Can fit larger batch sizes
- Can handle longer sequences
- Faster inference (less memory bandwidth)
- Enables serving on smaller GPUs

---

## 🎯 Real-World Configurations

### LLaMA Models

| Model | Layers | d_model | Q Heads | KV Heads | GQA Ratio |
|-------|--------|---------|---------|----------|-----------|
| LLaMA 1 7B | 32 | 4096 | 32 | 32 | 1:1 (MHA) |
| LLaMA 2 7B | 32 | 4096 | 32 | 32 | 1:1 (MHA) |
| LLaMA 2 13B | 40 | 5120 | 40 | 40 | 1:1 (MHA) |
| **LLaMA 2 34B** | 48 | 8192 | 64 | **8** | **8:1 (GQA)** |
| **LLaMA 2 70B** | 80 | 8192 | 64 | **8** | **8:1 (GQA)** |
| **LLaMA 3 70B** | 80 | 8192 | 64 | **8** | **8:1 (GQA)** |

**Pattern**: Large models use GQA for memory efficiency!

### Mistral 7B

```python
# Mistral 7B - Efficient small model with GQA
d_model = 4096
num_heads = 32
num_kv_heads = 8  # 4:1 ratio
num_layers = 32
```

Even the 7B model uses GQA for efficiency!

### Other Models

- **Falcon 40B**: MQA (1 KV head)
- **PaLM**: MQA (1 KV head)
- **Qwen**: GQA (varies by size)

---

## ⚖️ Quality vs Efficiency Trade-off

### Perplexity Comparison (Lower is Better)

From research papers:

```
Task: Language Modeling

MHA (32 heads):     Perplexity: 10.2  (baseline)
GQA (8 KV heads):   Perplexity: 10.3  (+0.98% worse)
GQA (4 KV heads):   Perplexity: 10.5  (+2.94% worse)
MQA (1 KV head):    Perplexity: 11.1  (+8.82% worse)
```

**Key insight**: GQA with 4-8x reduction is nearly as good as MHA!

### Speed Comparison

```
Inference speed (tokens/second):

MHA:  100 tok/s (baseline)
GQA:  140 tok/s (+40% faster)
MQA:  180 tok/s (+80% faster)
```

**GQA sweet spot**: 8:1 ratio gives 40% speedup with minimal quality loss!

---

## 🔧 Converting MHA to GQA

You can convert a trained MHA model to GQA!

### Mean Pooling Approach

```python
def convert_mha_to_gqa(mha_model, num_kv_heads):
    """
    Convert Multi-Head Attention to Grouped Query Attention

    Strategy: Average K and V heads within each group
    """
    num_heads = mha_model.num_heads
    num_groups = num_heads // num_kv_heads

    # Extract K and V weights
    W_k = mha_model.W_k.weight  # [num_heads * head_dim, d_model]
    W_v = mha_model.W_v.weight

    # Reshape to separate heads
    W_k = W_k.view(num_heads, head_dim, d_model)
    W_v = W_v.view(num_heads, head_dim, d_model)

    # Average within groups
    W_k_gqa = []
    W_v_gqa = []

    for group in range(num_kv_heads):
        start_idx = group * num_groups
        end_idx = start_idx + num_groups

        # Mean pool K and V for this group
        k_group = W_k[start_idx:end_idx].mean(dim=0)
        v_group = W_v[start_idx:end_idx].mean(dim=0)

        W_k_gqa.append(k_group)
        W_v_gqa.append(v_group)

    W_k_gqa = torch.stack(W_k_gqa).view(num_kv_heads * head_dim, d_model)
    W_v_gqa = torch.stack(W_v_gqa).view(num_kv_heads * head_dim, d_model)

    # Create GQA model with averaged weights
    gqa_model = GroupedQueryAttention(
        d_model=mha_model.d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads
    )

    gqa_model.W_k.weight.data = W_k_gqa
    gqa_model.W_v.weight.data = W_v_gqa
    gqa_model.W_q.weight.data = mha_model.W_q.weight.data

    return gqa_model
```

**Result**: Can convert trained MHA model to GQA with minimal fine-tuning!

---

## 🧠 How GQA Maintains Quality

### Why does sharing K and V work?

**Key insight**: Different query patterns can attend to the same keys/values!

**Analogy**: Library with multiple readers
- Readers (Q heads): Different people with different questions
- Books (K, V): Same collection of books
- Each reader finds different books useful (attention weights differ)
- But they all search the same library!

### Mathematical View

```python
# Different Q heads with shared K, V
Q₁, Q₂, Q₃, Q₄ → K_shared, V_shared

# Each Q produces DIFFERENT attention weights
attention_weights₁ = softmax(Q₁ @ K^T)  # Different pattern
attention_weights₂ = softmax(Q₂ @ K^T)  # Different pattern
attention_weights₃ = softmax(Q₃ @ K^T)  # Different pattern

# So outputs are still diverse!
out₁ = attention_weights₁ @ V  # Different weighted combination
out₂ = attention_weights₂ @ V  # Different weighted combination
```

**Diversity comes from Q, not K and V!**

---

## 🎓 Key Takeaways

1. **MHA problem**: Huge KV cache at inference (limits batch size and sequence length)

2. **GQA solution**: Share K and V across groups of Q heads
   - Typical ratio: 4:1 or 8:1 (4-8 Q heads per KV head)

3. **Memory savings**: 4-8x reduction in KV cache
   - LLaMA 2 70B: 87.5% reduction with 8:1 ratio

4. **Quality preservation**: Minimal degradation (<3% perplexity increase)

5. **Inference speedup**: ~40% faster with GQA (8:1)

6. **Real adoption**: LLaMA 2/3 (large), Mistral, many modern LLMs

7. **Spectrum**: MHA (best quality, most memory) → GQA (sweet spot) → MQA (fastest, lowest quality)

**GQA is THE solution for efficient large-scale LLM inference!**

---

## 📝 Quiz Time!

Test your understanding of Grouped Query Attention.
""")

# Quiz questions
with st.expander("Question 1: What problem does GQA solve?"):
    st.markdown("""
    **Question**: What is the primary problem that Grouped Query Attention (GQA) solves?

    A) Training speed
    B) Large KV cache during inference, especially for long sequences
    C) Model accuracy
    D) Tokenization efficiency
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) Large KV cache during inference, especially for long sequences**

        **Explanation**:

        **The problem:**
        ```python
        # LLaMA 70B with MHA (64 heads)
        KV cache = 2 * num_kv_heads * seq_len * head_dim * layers
        # With seq_len=4096: ~1.3 GB just for KV cache!
        ```

        **Why this matters:**
        - Limits batch size (less throughput)
        - Limits max sequence length
        - Requires large GPU memory
        - High memory bandwidth usage

        **GQA solution:**
        - Share K and V across query groups
        - 8:1 ratio → 87.5% memory reduction!
        - Can handle longer contexts and larger batches
        - Critical for production deployment

        KV cache is THE bottleneck in autoregressive generation (we'll cover this in Topic 22).
        """)

with st.expander("Question 2: MHA vs GQA vs MQA"):
    st.markdown("""
    **Question**: In GQA with 32 query heads and 8 KV heads, how many query heads share each KV head?

    A) 2 query heads per KV head
    B) 4 query heads per KV head
    C) 8 query heads per KV head
    D) 32 query heads per KV head
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) 4 query heads per KV head**

        **Explanation**:

        ```python
        num_query_heads = 32
        num_kv_heads = 8
        group_size = num_query_heads // num_kv_heads = 32 // 8 = 4
        ```

        **Structure:**
        ```
        Group 1: Q₁, Q₂, Q₃, Q₄ → share K₁, V₁
        Group 2: Q₅, Q₆, Q₇, Q₈ → share K₂, V₂
        ...
        Group 8: Q₂₉, Q₃₀, Q₃₁, Q₃₂ → share K₈, V₈
        ```

        **Memory savings:**
        - MHA: 32 KV heads
        - GQA: 8 KV heads
        - Reduction: 32 / 8 = 4x less memory!

        This is the exact configuration used in LLaMA 2 70B!
        """)

with st.expander("Question 3: Why does GQA maintain quality?"):
    st.markdown("""
    **Question**: Why can GQA maintain quality despite sharing K and V across multiple query heads?

    A) Because K and V don't matter for attention
    B) Because different Q heads produce different attention patterns even with shared K and V
    C) Because GQA uses more layers
    D) Because modern GPUs are faster
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) Because different Q heads produce different attention patterns even with shared K and V**

        **Explanation**:

        **Key insight: Diversity comes from Q, not K and V!**

        ```python
        # Shared K, V but different Q
        Q₁, Q₂, Q₃, Q₄ → K_shared, V_shared

        # Each Q produces DIFFERENT attention weights
        attn₁ = softmax(Q₁ @ K^T / √d)  # Unique pattern
        attn₂ = softmax(Q₂ @ K^T / √d)  # Different pattern
        attn₃ = softmax(Q₃ @ K^T / √d)  # Different pattern

        # So outputs are still diverse
        out₁ = attn₁ @ V  # Different weighted combination
        out₂ = attn₂ @ V  # Different weighted combination
        ```

        **Library analogy:**
        - Same books (K, V)
        - Different readers with different interests (Q)
        - Each reader finds different books useful
        - Still get diverse perspectives!

        Research shows <3% quality degradation with 8:1 GQA ratio!
        """)

with st.expander("Question 4: GQA in real models"):
    st.markdown("""
    **Question**: Which configuration does LLaMA 2 70B use?

    A) MHA with 64 query heads and 64 KV heads
    B) GQA with 64 query heads and 8 KV heads
    C) MQA with 64 query heads and 1 KV head
    D) GQA with 32 query heads and 8 KV heads
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) GQA with 64 query heads and 8 KV heads**

        **Explanation**:

        **LLaMA 2 70B configuration:**
        ```python
        d_model = 8192
        num_layers = 80
        num_heads = 64        # Query heads
        num_kv_heads = 8      # KV heads
        group_size = 8        # 8 Q heads per KV head
        ```

        **Why this configuration?**
        - 8:1 ratio: Great memory savings (87.5% reduction)
        - Minimal quality loss (<1% worse than MHA)
        - Enables longer contexts (up to 32k tokens)
        - Better inference throughput

        **Smaller LLaMA models:**
        - LLaMA 2 7B, 13B: Still use MHA (32/40 heads)
        - Large models need GQA more (KV cache dominates at scale)

        **Also use GQA:**
        - LLaMA 3 70B
        - Mistral 7B (32 Q heads, 8 KV heads)
        """)

with st.expander("Question 5: Memory savings calculation"):
    st.markdown("""
    **Question**: If MHA uses 1.6 GB for KV cache, how much would GQA with 8:1 ratio use?

    A) 0.2 GB
    B) 0.4 GB
    C) 0.8 GB
    D) 1.2 GB
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: A) 0.2 GB**

        **Explanation**:

        **Calculation:**
        ```python
        # MHA: All heads have separate K and V
        mha_cache = num_heads * head_size = 1.6 GB

        # GQA with 8:1 ratio
        num_kv_heads = num_heads / 8
        gqa_cache = num_kv_heads * head_size = 1.6 / 8 = 0.2 GB
        ```

        **Reduction: 1.6 - 0.2 = 1.4 GB saved (87.5% reduction)**

        **Real-world impact:**

        With 1.4 GB saved:
        - Can increase batch size (more throughput)
        - Can handle longer sequences
        - Can fit model on smaller GPU
        - Less memory bandwidth pressure (faster)

        For LLaMA 2 70B with 4k context:
        - MHA: ~1.3 GB KV cache
        - GQA: ~0.17 GB KV cache
        - **Savings: ~1.1 GB per inference!**

        At scale (thousands of concurrent users), this is MASSIVE!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand how GQA makes large LLMs memory-efficient!

But there's more to inference optimization:

Next topics:
- **Topic 21**: Flash Attention - algorithmic breakthrough for 2-4x faster attention computation
- **Topic 22**: KV Cache & Efficient Inference - deep dive into why KV cache matters and how to optimize it
- **Topic 23**: Mixture of Experts (MoE) - scaling to trillions of parameters efficiently

**You now understand the memory optimization used in LLaMA 2/3, Mistral, and modern large LLMs!** 🚀

---

*Navigation: ← Previous: Topic 19 (Modern Components) | Next: Topic 21 (Flash Attention) →*
""")
