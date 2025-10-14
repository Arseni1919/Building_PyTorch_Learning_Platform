import streamlit as st

st.title("⚡ Topic 15: Multi-Head Attention")

st.markdown("""
---

## 🎯 Why One Attention Head Isn't Enough

You just learned about attention - the mechanism that revolutionized AI. But here's the thing: **one attention mechanism isn't enough**!

Think about reading a sentence. You simultaneously pay attention to:
- **Syntax**: Subject-verb agreement, grammar structure
- **Semantics**: Word meanings and relationships
- **Context**: References, pronouns, dependencies
- **Style**: Tone, sentiment, formality

One attention head would have to learn ALL of these patterns at once. That's like asking one person to be an expert in grammar, meaning, context, AND style simultaneously!

**The solution?** Use **multiple attention heads** in parallel, each specializing in different patterns.

This is **Multi-Head Attention** - and it's used in EVERY modern LLM:
- GPT-4: Uses many attention heads per layer
- LLaMA 3: 32-128 heads depending on model size
- Claude: Multi-head attention architecture
- BERT: 12 heads in base, 16 in large

---

## 💡 The Core Idea: Multiple Representation Subspaces

### Single-Head Attention Problem

With single-head attention, the model tries to capture everything in one attention pattern:

```python
# Single head: tries to learn everything at once
Q, K, V = linear_projection(input)
output = attention(Q, K, V)  # One attention pattern
```

**Limitations:**
- ❌ Can't specialize in different types of relationships
- ❌ May miss important patterns while focusing on others
- ❌ Limited capacity to represent complex dependencies

### Multi-Head Attention Solution

Split the model dimension into multiple "heads", each learning different patterns:

```python
# Multiple heads: each specializes in different patterns
for i in range(num_heads):
    Q_i, K_i, V_i = linear_projection_i(input)  # Different projections
    head_i = attention(Q_i, K_i, V_i)           # Different patterns

output = concat(head_1, head_2, ..., head_n)  # Combine all insights
```

**Benefits:**
- ✅ Each head specializes in different relationship types
- ✅ Parallel processing of multiple patterns
- ✅ Richer, more comprehensive representation
- ✅ Better gradient flow during training

---

## 🧠 How Multi-Head Attention Works

### Architecture Overview

```
Input: [batch, seq_len, d_model]
         ↓
Split into h heads: [batch, seq_len, d_model] → [batch, h, seq_len, d_k]
         ↓
   ┌──────────────────┐
   │  Head 1 (Q,K,V)  │ → Attention → Output 1
   │  Head 2 (Q,K,V)  │ → Attention → Output 2
   │       ...        │      ...          ...
   │  Head h (Q,K,V)  │ → Attention → Output h
   └──────────────────┘
         ↓
Concatenate: [batch, seq_len, h * d_v]
         ↓
Linear projection: [batch, seq_len, d_model]
         ↓
Output: [batch, seq_len, d_model]
```

### Key Insight: Dimension Splitting

Instead of one large attention with dimension `d_model`, we use `h` smaller attentions:

```python
d_model = 512        # Total dimension
num_heads = 8        # Number of heads
d_k = d_model // num_heads = 64  # Dimension per head
```

**Why this works:**
- Same total parameters as single large attention
- Each head gets its own subspace to explore
- Different heads learn complementary patterns

---

## 🔢 The Mathematics

### Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
```

Let's break it down:

### Step 1: Linear Projections for Each Head

Each head has its own weight matrices:

```python
# For head i:
Q_i = Q @ W^Q_i  # [batch, seq_len, d_model] @ [d_model, d_k] → [batch, seq_len, d_k]
K_i = K @ W^K_i  # [batch, seq_len, d_model] @ [d_model, d_k] → [batch, seq_len, d_k]
V_i = V @ W^V_i  # [batch, seq_len, d_model] @ [d_model, d_v] → [batch, seq_len, d_v]
```

**Why separate projections?**
- Each head learns its own transformation
- Different heads can focus on different aspects
- More expressive than shared projections

### Step 2: Attention for Each Head

```python
head_i = Attention(Q_i, K_i, V_i)
       = softmax(Q_i K_i^T / √d_k) V_i
```

This is the same scaled dot-product attention from Topic 14!

### Step 3: Concatenate All Heads

```python
multi_head_output = Concat(head_1, head_2, ..., head_h)
# [batch, seq_len, h * d_v]
```

Combine all the different perspectives.

### Step 4: Final Linear Projection

```python
output = multi_head_output @ W^O
# [batch, seq_len, h * d_v] @ [h * d_v, d_model] → [batch, seq_len, d_model]
```

Mix information from all heads together.

---

## 💻 Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from 'Attention is All You Need'

    This is used in GPT, BERT, LLaMA, Claude, and EVERY modern LLM!
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: Total dimension of the model (e.g., 512)
            num_heads: Number of attention heads (e.g., 8)
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V (all heads at once for efficiency)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Inverse of split_heads
        Args:
            x: [batch_size, num_heads, seq_len, d_k]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: Optional [batch_size, 1, seq_len_q, seq_len_k]

        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 1. Linear projections for all heads at once
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_v, d_model]

        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len_q, d_k]
        K = self.split_heads(K)  # [batch, num_heads, seq_len_k, d_k]
        V = self.split_heads(V)  # [batch, num_heads, seq_len_v, d_k]

        # 3. Scaled dot-product attention for all heads in parallel
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [batch, num_heads, seq_len_q, seq_len_k]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 4. Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        # [batch, num_heads, seq_len_q, d_k]

        # 5. Concatenate heads
        output = self.combine_heads(attention_output)
        # [batch, seq_len_q, d_model]

        # 6. Final linear projection
        output = self.W_o(output)

        return output, attention_weights


# Example usage
batch_size = 2
seq_len = 10
d_model = 512
num_heads = 8

# Input sequence
x = torch.randn(batch_size, seq_len, d_model)

# Create multi-head attention
mha = MultiHeadAttention(d_model, num_heads)

# Apply multi-head attention (self-attention)
output, attention_weights = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"\\nNumber of heads: {num_heads}")
print(f"Dimension per head (d_k): {d_model // num_heads}")
```

**Output:**
```
Input shape: torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
Attention weights shape: torch.Size([2, 8, 10, 10])

Number of heads: 8
Dimension per head (d_k): 64
```

---

## 🎨 What Do Different Heads Learn?

This is where it gets fascinating! Different heads specialize in different patterns:

### Example: Analyzing BERT Heads

Research has shown that in BERT, different heads learn:

**Head 1**: Syntax and grammar
```
"The cat [sits] on the mat"
       ↑
Attends to subject "cat"
```

**Head 2**: Positional relationships
```
"The [cat] sits on the mat"
      ↑
Attends to nearby words "The" and "sits"
```

**Head 3**: Semantic similarity
```
"The [cat] sits on the mat"
      ↑
Attends to related concepts
```

**Head 4**: Long-range dependencies
```
"The cat, which was black, [sits] on the mat"
                            ↑
Attends to distant subject "cat"
```

### Visualization Example

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_multi_head_attention(attention_weights, tokens, num_heads_to_show=4):
    """
    Visualize attention patterns from different heads
    """
    # attention_weights: [batch, num_heads, seq_len, seq_len]
    weights = attention_weights[0].detach().numpy()  # First sample

    fig, axes = plt.subplots(1, num_heads_to_show, figsize=(16, 4))

    for i in range(num_heads_to_show):
        ax = axes[i]
        im = ax.imshow(weights[i], cmap='viridis', aspect='auto')

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticklabels(tokens)

        ax.set_title(f"Head {i+1}")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig

# Example
tokens = ["The", "cat", "sat", "on", "mat"]
# Assuming we have attention_weights from the model
# fig = visualize_multi_head_attention(attention_weights, tokens)
```

**What you'd see:**
- Different heads show different bright spots
- Some heads focus on adjacent words (syntax)
- Some heads focus on specific relationships (semantics)
- Some heads create long-range connections

---

## 🔍 Parallel Processing: The Key Advantage

### Single-Head Attention: Sequential Bottleneck

```python
# Single head: One pattern at a time
output_1 = attention(input)  # Learn one pattern
```

### Multi-Head Attention: Parallel Exploration

```python
# Multi-head: Multiple patterns simultaneously
head_1 = attention_1(input)  # Syntax patterns
head_2 = attention_2(input)  # Semantic patterns
head_3 = attention_3(input)  # Positional patterns
...
output = combine(head_1, head_2, head_3, ...)  # Rich representation
```

**Benefits:**
1. **Richer representations**: Multiple perspectives on the same input
2. **Specialization**: Each head can focus on what it does best
3. **Robustness**: If one head fails, others compensate
4. **Parallel computation**: All heads computed simultaneously on GPU

---

## 🎯 Head Dimension Calculation

Critical design choice: How to split dimensions?

### Common Configurations

```python
# GPT-2 Small
d_model = 768
num_heads = 12
d_k = 768 // 12 = 64

# GPT-2 Medium
d_model = 1024
num_heads = 16
d_k = 1024 // 16 = 64

# GPT-2 Large
d_model = 1280
num_heads = 20
d_k = 1280 // 20 = 64

# LLaMA 7B
d_model = 4096
num_heads = 32
d_k = 4096 // 32 = 128
```

**Pattern**: `d_k` is typically 64 or 128!

**Why?**
- Too small (d_k = 16): Limited expressiveness per head
- Too large (d_k = 256): Fewer heads, less diversity
- Sweet spot (d_k = 64-128): Good balance

---

## 🚀 Used in EVERY Modern LLM

### GPT Architecture (Decoder-only)
```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)  # ← HERE!
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Multi-head self-attention
        x = x + self.ffn(self.ln2(x))
        return x
```

### BERT Architecture (Encoder-only)
```python
class BERTBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        self.attn = MultiHeadAttention(d_model, num_heads)  # ← HERE!
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))  # Multi-head self-attention
        x = self.ln2(x + self.ffn(x))
        return x
```

### LLaMA 3 Configuration
```python
# LLaMA 3 8B
d_model = 4096
num_heads = 32       # Multi-head attention
num_kv_heads = 8     # Grouped Query Attention (Topic 20!)
layers = 32

# LLaMA 3 70B
d_model = 8192
num_heads = 64
num_kv_heads = 8
layers = 80
```

---

## 🧩 Concatenation vs Addition: Why Concat?

You might wonder: Why concatenate heads instead of adding them?

### If We Added:
```python
# Bad: Adding heads (loses information)
head_1 = attention_1(input)  # [batch, seq_len, d_model]
head_2 = attention_2(input)  # [batch, seq_len, d_model]
output = head_1 + head_2     # [batch, seq_len, d_model]
```

**Problem**: Addition mixes information, destroying specialization!

### Concatenation (Correct):
```python
# Good: Concatenating heads (preserves information)
head_1 = attention_1(input)  # [batch, seq_len, d_k]
head_2 = attention_2(input)  # [batch, seq_len, d_k]
output = concat(head_1, head_2)  # [batch, seq_len, 2*d_k]
```

**Benefit**: Each head's output is preserved in a separate subspace!

Then the output projection `W_o` learns how to combine them.

---

## 💡 Key Insights

### Why Multi-Head Works

1. **Ensemble effect**: Like having multiple experts vote
2. **Representation diversity**: Different subspaces capture different patterns
3. **Gradient flow**: More paths for gradients during backprop
4. **Computational efficiency**: Parallel processing on GPU

### Implementation Tricks

1. **Efficient batching**: Compute all heads at once with reshape operations
2. **Shared projection**: One linear layer for all heads instead of separate layers
3. **Fused kernels**: Modern implementations fuse operations for speed

---

## 🎓 Key Takeaways

1. **One head isn't enough**: Different patterns require different attention mechanisms
2. **Parallel specialization**: Multiple heads learn complementary patterns
3. **Dimension splitting**: `d_model` split into `num_heads` heads of size `d_k`
4. **Concatenation preserves diversity**: Each head's output kept separate
5. **Universal adoption**: Every modern LLM uses multi-head attention
6. **Typical configurations**: d_k = 64 or 128, num_heads = 8-64+ depending on model size

**Multi-head attention is THE architecture that powers GPT-4, Claude 3, LLaMA, and every modern LLM!**

---

## 📝 Quiz Time!

Test your understanding of multi-head attention.
""")

# Quiz questions
with st.expander("Question 1: Why do we need multiple heads?"):
    st.markdown("""
    **Question**: What is the PRIMARY reason for using multiple attention heads instead of one large head?

    A) Multiple heads train faster
    B) Multiple heads use less memory
    C) Multiple heads can learn different types of relationships in parallel
    D) Multiple heads require fewer parameters
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: C) Multiple heads can learn different types of relationships in parallel**

        **Explanation**: Each attention head can specialize in different patterns:
        - Head 1: Syntax and grammar
        - Head 2: Semantic relationships
        - Head 3: Long-range dependencies
        - Head 4: Positional patterns

        A single head would have to learn ALL patterns simultaneously, which is much harder. Multiple heads provide richer, more diverse representations.

        Note: Multi-head attention actually uses the SAME number of parameters as a single large head (d_model dimension is split across heads).
        """)

with st.expander("Question 2: Dimension calculation"):
    st.markdown("""
    **Question**: If d_model = 768 and num_heads = 12, what is d_k (dimension per head)?

    A) 32
    B) 64
    C) 128
    D) 768
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) 64**

        **Explanation**:
        ```python
        d_k = d_model // num_heads
        d_k = 768 // 12 = 64
        ```

        This is the exact configuration used in GPT-2 Small and BERT Base!

        The total dimension (768) is split evenly across all heads (12), giving each head a 64-dimensional subspace to work in.
        """)

with st.expander("Question 3: Why concatenate instead of add?"):
    st.markdown("""
    **Question**: Why do we concatenate head outputs instead of adding them?

    A) Concatenation is faster to compute
    B) Concatenation preserves each head's specialized information
    C) Addition would change the output dimension
    D) Concatenation uses less memory
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) Concatenation preserves each head's specialized information**

        **Explanation**:

        **If we added:**
        ```python
        head_1 + head_2 + head_3  # Information gets mixed, loses specialization
        ```

        **By concatenating:**
        ```python
        concat(head_1, head_2, head_3)  # Each head's output preserved separately
        ```

        Concatenation keeps each head's specialized patterns in separate subspaces. Then the output projection W_o learns how to optimally combine them.

        This is crucial for maintaining the diversity that makes multi-head attention powerful!
        """)

with st.expander("Question 4: Multi-head attention shapes"):
    st.markdown("""
    **Question**: Given input shape [batch=2, seq_len=10, d_model=512] and num_heads=8, what is the shape AFTER splitting heads but BEFORE attention?

    A) [2, 10, 512]
    B) [2, 8, 10, 64]
    C) [2, 10, 8, 64]
    D) [2, 8, 64, 10]
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) [2, 8, 10, 64]**

        **Explanation**:

        Transformation steps:
        ```python
        # 1. Input
        x: [batch=2, seq_len=10, d_model=512]

        # 2. Linear projection (no shape change)
        Q: [2, 10, 512]

        # 3. Reshape to split heads
        Q_reshaped: [2, 10, num_heads=8, d_k=64]

        # 4. Transpose to move heads dimension
        Q_split: [2, num_heads=8, seq_len=10, d_k=64]
        ```

        This shape allows us to compute attention for all heads in parallel!

        d_k = 512 // 8 = 64
        """)

with st.expander("Question 5: Real-world configurations"):
    st.markdown("""
    **Question**: Which of the following is a typical configuration for modern LLMs?

    A) d_k = 16 (very small heads)
    B) d_k = 64-128 (moderate-sized heads)
    C) d_k = 512 (very large heads)
    D) d_k varies randomly across different heads
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: B) d_k = 64-128 (moderate-sized heads)**

        **Explanation**:

        Real-world configurations:
        ```python
        # GPT-2 Small
        d_model = 768, num_heads = 12, d_k = 64

        # BERT Base
        d_model = 768, num_heads = 12, d_k = 64

        # GPT-2 Large
        d_model = 1280, num_heads = 20, d_k = 64

        # LLaMA 7B
        d_model = 4096, num_heads = 32, d_k = 128
        ```

        **Why 64-128 is the sweet spot:**
        - Too small (16): Limited expressiveness per head
        - Too large (512): Fewer heads, less diversity
        - Just right (64-128): Good balance between head capacity and diversity

        This pattern holds across GPT, BERT, LLaMA, and virtually all modern transformers!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand multi-head attention - the parallel processing powerhouse of modern LLMs!

But there's one problem we haven't solved: **Transformers have no sense of word order!**

"cat sat mat" and "mat sat cat" would look identical to attention. That's where **positional encoding** comes in.

Next topics:
- **Topic 16**: Positional Encoding (teaching transformers about sequence order)
- **Topic 17**: RoPE (modern positional encoding used in LLaMA)
- **Topic 18**: Complete Transformer Architecture

**You now understand why GPT-4 uses dozens of attention heads per layer!** 🚀

---

*Navigation: ← Previous: Topic 14 (Attention Mechanism) | Next: Topic 16 (Positional Encoding) →*
""")
