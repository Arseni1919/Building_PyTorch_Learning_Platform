import streamlit as st

st.title("⚡ Topic 16: Positional Encoding")

st.markdown("""
---

## 🎯 The Problem: Transformers Are Order-Blind!

You've learned about attention and multi-head attention - the mechanisms that power modern LLMs. But there's a **critical problem**:

**Attention has no concept of position or order!**

Consider these two sentences:
1. "The dog chased the cat"
2. "The cat chased the dog"

To attention, these are **identical** (just rearranged words). The attention mechanism computes similarity between words but **doesn't know which word comes first**.

```python
# Attention sees both sentences as the same set of words
sentence_1 = {"The", "dog", "chased", "the", "cat"}
sentence_2 = {"The", "cat", "chased", "the", "dog"}
# Same set! But completely different meanings!
```

**This is a HUGE problem** because word order is crucial in language:
- "I didn't say I love you" ≠ "I love you didn't say I"
- "Not bad" ≠ "Bad not"
- "Man bites dog" ≠ "Dog bites man"

**The solution?** **Positional Encoding** - injecting position information into the model.

---

## 💡 Why Position Matters in Language

### RNNs Had Built-in Position Awareness

```python
# RNNs process sequentially, so position is implicit
h_1 = RNN(word_1, h_0)        # First word
h_2 = RNN(word_2, h_1)        # Second word (knows it comes after word_1)
h_3 = RNN(word_3, h_2)        # Third word (knows it comes after word_2)
```

Position is encoded in the sequential processing!

### Transformers Process Everything in Parallel

```python
# Transformers see all words at once (no sequential order)
word_1, word_2, word_3 → Attention → output
# Without position info, word_1 and word_3 are indistinguishable positions!
```

**Solution needed**: Explicitly add position information to word embeddings.

---

## 🧩 Approaches to Positional Encoding

There are several ways to add position information:

### 1. Learned Positional Embeddings (BERT, GPT-2)

Simple idea: Learn position embeddings like word embeddings!

```python
# Learned approach
position_embedding = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(seq_len)
pos_encoded = position_embedding(positions)

output = word_embedding + pos_encoded
```

**Pros:**
- ✅ Simple to implement
- ✅ Flexible - model learns what works best

**Cons:**
- ❌ Fixed maximum sequence length
- ❌ Doesn't generalize to longer sequences
- ❌ Each position is independent (position 5 doesn't know it's close to position 6)

### 2. Sinusoidal Positional Encoding (Original Transformer)

Clever idea: Use mathematical functions (sine and cosine) to encode position!

```python
# Sinusoidal approach (we'll implement this!)
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Pros:**
- ✅ No learned parameters
- ✅ Can generalize to longer sequences
- ✅ Mathematical properties (relative positions)
- ✅ Smooth, continuous representation

**Cons:**
- ❌ Absolute positions (position 5 encoded the same in all contexts)
- ❌ Doesn't adapt to data

**This is what we'll focus on - the original transformer approach!**

### 3. Relative Positional Encoding

Modern idea: Encode relative distances instead of absolute positions!

This led to **RoPE** (Rotary Position Embeddings) - Topic 17!

---

## 🔢 Sinusoidal Positional Encoding: The Math

### The Formula

For position `pos` and dimension `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Let's decode this:

### Why Sine and Cosine?

1. **Periodic functions**: Repeat in a pattern (useful for similar positions)
2. **Different frequencies**: Low dimensions change slowly, high dimensions change quickly
3. **Unique encodings**: Each position gets a unique pattern
4. **Smooth**: Close positions have similar encodings

### Intuition: Frequency Bands

Think of it like a **binary clock** but with sine waves:

```
Position encodings across dimensions:

Dimension 0-1:   sin/cos with wavelength ~60k (very slow oscillation)
Dimension 2-3:   sin/cos with wavelength ~6k
Dimension 4-5:   sin/cos with wavelength ~600
...
Dimension 510-511: sin/cos with wavelength ~10 (very fast oscillation)
```

**Low dimensions**: Encode "rough" position (e.g., "beginning", "middle", "end")
**High dimensions**: Encode "precise" position (e.g., exact token index)

### Visual Example

```
Position 0:  [sin(0/10000), cos(0/10000), sin(0/100), cos(0/100), ...]
Position 1:  [sin(1/10000), cos(1/10000), sin(1/100), cos(1/100), ...]
Position 2:  [sin(2/10000), cos(2/10000), sin(2/100), cos(2/100), ...]
...
```

Each position gets a unique "fingerprint" of sine and cosine values!

---

## 💻 Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from 'Attention is All You Need'

    Adds position information to token embeddings using sine and cosine functions.
    """

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model (e.g., 512)
            max_seq_len: Maximum sequence length to pre-compute (e.g., 5000)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)

        # Create position indices [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Shape: [max_seq_len, 1]

        # Create dimension indices and compute division term
        # div_term = 10000^(2i/d_model) for i in [0, 1, 2, ..., d_model/2-1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Shape: [d_model/2]

        # Apply sine to even indices in the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the positional encoding
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [max_seq_len, d_model] → [1, max_seq_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]

        Returns:
            x: Embeddings with positional encoding added [batch_size, seq_len, d_model]
        """
        # Add positional encoding (broadcasting across batch dimension)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Example usage
batch_size = 2
seq_len = 10
d_model = 512

# Create random embeddings (from embedding layer)
embeddings = torch.randn(batch_size, seq_len, d_model)

# Create positional encoding
pos_encoding = PositionalEncoding(d_model)

# Add positional information
output = pos_encoding(embeddings)

print(f"Input embeddings shape: {embeddings.shape}")
print(f"Output (with position info) shape: {output.shape}")
print(f"\\nFirst 5 positions, first 8 dimensions of PE:")
print(pos_encoding.pe[0, :5, :8])
```

**Output:**
```
Input embeddings shape: torch.Size([2, 10, 512])
Output (with position info) shape: torch.Size([2, 10, 512])

First 5 positions, first 8 dimensions of PE:
tensor([[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
        [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000,  0.0001,  1.0000],
        [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000,  0.0002,  1.0000],
        [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  0.9999,  0.0003,  1.0000],
        [-0.7568, -0.6536,  0.1846,  0.9828,  0.0086,  0.9999,  0.0004,  1.0000]])
```

Notice the patterns:
- Position 0: All cosines are 1, all sines are 0
- Each position has a unique combination of values
- Low dimensions (0-1) change quickly
- High dimensions (6-7) change slowly

---

## 📊 Visualizing Positional Encodings

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_positional_encoding(d_model=512, max_seq_len=100):
    """
    Visualize sinusoidal positional encoding patterns
    """
    # Create positional encoding
    pe = PositionalEncoding(d_model, max_seq_len)

    # Extract the encoding matrix
    encoding = pe.pe[0, :max_seq_len, :].numpy()

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. Heatmap of positional encodings
    im = axes[0].imshow(encoding.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Dimension')
    axes[0].set_title('Positional Encoding Heatmap')
    plt.colorbar(im, ax=axes[0])

    # 2. Specific dimensions over positions
    axes[1].plot(encoding[:, 0], label='Dimension 0 (sin, slow)')
    axes[1].plot(encoding[:, 1], label='Dimension 1 (cos, slow)')
    axes[1].plot(encoding[:, 4], label='Dimension 4 (sin, medium)')
    axes[1].plot(encoding[:, 5], label='Dimension 5 (cos, medium)')
    axes[1].plot(encoding[:, 128], label='Dimension 128 (sin, fast)')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Value')
    axes[1].set_title('PE Values Across Positions (Selected Dimensions)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Uncomment to visualize:
# fig = visualize_positional_encoding()
# plt.show()
```

**What you'd see:**
- Horizontal stripes in the heatmap (different frequencies for different dimensions)
- Low dimensions: Long wavelength (slow oscillation)
- High dimensions: Short wavelength (fast oscillation)
- Each position has a unique pattern across all dimensions

---

## 🔍 Mathematical Properties

### Property 1: Unique Encoding for Each Position

Each position gets a unique vector:
```python
PE(0) ≠ PE(1) ≠ PE(2) ≠ ...
```

No two positions have the same encoding!

### Property 2: Relative Position Encoding

The key insight: For any fixed offset k, PE(pos+k) can be represented as a **linear function** of PE(pos).

```python
PE(pos + k) = M_k @ PE(pos)  # For some matrix M_k
```

**Why this matters**: The model can learn to attend to relative positions (e.g., "the word 3 positions ago") rather than just absolute positions.

### Property 3: Bounded Values

```python
-1 ≤ PE(pos, i) ≤ 1  # All values between -1 and 1
```

Sine and cosine are bounded, so positional encodings don't explode for large positions.

### Property 4: Generalizes to Unseen Lengths

Because it's a mathematical function, it works for any sequence length:
```python
# Trained on sequences up to length 512
# Can still encode position 1000!
PE(1000) = [sin(1000/10000^...), cos(1000/10000^...), ...]
```

---

## 🎯 How It's Used in Transformers

### Complete Embedding Process

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)

    def forward(self, x):
        # x: [batch, seq_len] - token indices

        # 1. Convert tokens to embeddings
        token_emb = self.token_embedding(x)  # [batch, seq_len, d_model]

        # 2. Add positional information
        output = self.position_encoding(token_emb)  # [batch, seq_len, d_model]

        return output


# Example
vocab_size = 10000
d_model = 512
max_seq_len = 5000

embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)

# Input: token indices
tokens = torch.randint(0, vocab_size, (2, 10))  # [batch=2, seq_len=10]

# Get embeddings with positional info
embedded = embedding(tokens)  # [2, 10, 512]

print(f"Input tokens shape: {tokens.shape}")
print(f"Embedded with position shape: {embedded.shape}")
```

**The flow:**
```
Token IDs → Token Embeddings → + Positional Encoding → To Transformer
[batch, seq_len] → [batch, seq_len, d_model] → [batch, seq_len, d_model]
```

---

## 🚀 Used in Modern LLMs

### Original Transformer (2017)
```python
# Sinusoidal positional encoding (what we just learned!)
output = token_embedding + sinusoidal_PE
```

### BERT (2018)
```python
# Learned positional embeddings
position_embedding = nn.Embedding(max_position=512, d_model=768)
output = token_embedding + position_embedding(positions)
```

### GPT-2 (2019)
```python
# Learned positional embeddings
position_embedding = nn.Embedding(max_position=1024, d_model=768)
output = token_embedding + position_embedding(positions)
```

### GPT-3 (2020)
```python
# Learned positional embeddings (longer sequences)
position_embedding = nn.Embedding(max_position=2048, d_model=12288)
output = token_embedding + position_embedding(positions)
```

### Modern Models: The Shift to RoPE

**LLaMA, GPT-Neo, PaLM, Mistral (2023-2025)**: Use **RoPE** (Rotary Position Embeddings)

Why the shift?
- ✅ Better extrapolation to longer sequences
- ✅ Relative position information
- ✅ Works better for long-context tasks

We'll cover RoPE in **Topic 17**!

---

## 📊 Limitations of Sinusoidal Encoding

### Limitation 1: Absolute Positions

Position 5 is always encoded the same way, regardless of context:
```python
PE(5) in "The cat sat on mat" == PE(5) in "I love machine learning"
```

**Problem**: Position 5 in a 10-word sentence is different from position 5 in a 1000-word document!

### Limitation 2: Extrapolation Quality

While theoretically it works for any length, in practice:
```python
# Trained on sequences ≤ 512 tokens
# Performance degrades at 2000+ tokens
```

### Limitation 3: No Adaptation to Data

Fixed mathematical function - can't learn from data:
```python
# Maybe some dimensions are more useful than others?
# Sinusoidal encoding can't adapt!
```

**These limitations led to the development of RoPE (Topic 17)!**

---

## 💡 Learned vs Sinusoidal: The Debate

### Sinusoidal (Original Transformer)
**Pros:**
- No extra parameters
- Generalizes to any length
- Mathematical interpretability

**Cons:**
- Doesn't adapt to data
- Absolute positions

### Learned (BERT, GPT-2)
**Pros:**
- Adapts to data
- Often works slightly better in practice
- Simpler to understand

**Cons:**
- Fixed max length
- More parameters
- Doesn't extrapolate well

**In practice**: Both work well! Modern research has moved to RoPE for best of both worlds.

---

## 🎓 Key Takeaways

1. **Attention is order-blind**: Without position info, "cat dog" = "dog cat"
2. **Positional encoding solution**: Add position information to embeddings
3. **Sinusoidal encoding**: Use sin/cos at different frequencies
4. **Unique fingerprints**: Each position gets a unique encoding pattern
5. **Mathematical properties**: Bounded, generalizable, enables relative position learning
6. **Evolution**: Sinusoidal → Learned → RoPE (modern standard)
7. **Universal need**: ALL transformers need some form of positional encoding

**Without positional encoding, transformers couldn't understand language order!**

---

## 📝 Quiz Time!

Test your understanding of positional encoding.
""")

# Quiz questions
with st.expander("Question 1: Why do transformers need positional encoding?"):
    st.markdown("""
    **Question**: Why do transformers need positional encoding while RNNs don't?

    A) Transformers have more parameters
    B) Transformers process sequences in parallel, so position isn't implicit
    C) RNNs are faster
    D) Transformers use attention
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) Transformers process sequences in parallel, so position isn't implicit**

        **Explanation**:

        **RNNs**: Sequential processing → position is implicit
        ```python
        h_1 = RNN(word_1, h_0)  # Knows it's first
        h_2 = RNN(word_2, h_1)  # Knows it comes after word_1
        ```

        **Transformers**: Parallel processing → need explicit position
        ```python
        # All words processed simultaneously
        attention(word_1, word_2, word_3)
        # Without position info, can't tell word_1 from word_3!
        ```

        Without positional encoding, "dog bites man" and "man bites dog" would be identical to a transformer!
        """)

with st.expander("Question 2: Sinusoidal encoding formula"):
    st.markdown("""
    **Question**: In sinusoidal positional encoding, why are both sine AND cosine used?

    A) To make computation faster
    B) To use even and odd dimensions separately
    C) To reduce memory usage
    D) To make the model deeper
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) To use even and odd dimensions separately**

        **Explanation**:

        ```python
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))  # Even dimensions (0, 2, 4, ...)
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  # Odd dimensions (1, 3, 5, ...)
        ```

        **Why both?**
        - Sine and cosine are orthogonal (provide different information)
        - Together they form a complete representation
        - Mathematical property: enables relative position encoding
        - Doubles the representational capacity

        Using both gives each position a richer, more unique encoding!
        """)

with st.expander("Question 3: Advantage of sinusoidal encoding"):
    st.markdown("""
    **Question**: What is the main advantage of sinusoidal positional encoding over learned embeddings?

    A) Faster training
    B) Fewer parameters and can generalize to unseen sequence lengths
    C) Better accuracy
    D) Easier to implement
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) Fewer parameters and can generalize to unseen sequence lengths**

        **Explanation**:

        **Sinusoidal (mathematical function):**
        - 0 learned parameters
        - Can encode ANY position (even 10,000 if trained on max 512)
        - Works for sequences longer than seen during training

        **Learned embeddings:**
        - max_seq_len × d_model parameters (e.g., 512 × 768 = 393k parameters)
        - Fixed maximum length
        - Can't handle sequences longer than max_position

        **Example:**
        ```python
        # Sinusoidal: Trained on length 512
        PE(1000) = sin(1000/10000^...), cos(...)  # Still works!

        # Learned: Trained on length 512
        position_embedding[1000]  # ERROR! Out of bounds!
        ```
        """)

with st.expander("Question 4: Frequency patterns"):
    st.markdown("""
    **Question**: In sinusoidal positional encoding, what changes across different dimensions?

    A) The amplitude of the sine/cosine waves
    B) The frequency/wavelength of the sine/cosine waves
    C) The phase of the sine/cosine waves
    D) The offset of the sine/cosine waves
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) The frequency/wavelength of the sine/cosine waves**

        **Explanation**:

        ```python
        # Low dimensions: Low frequency (long wavelength, slow oscillation)
        PE(pos, 0) = sin(pos / 10000^0) = sin(pos)

        # Middle dimensions: Medium frequency
        PE(pos, 256) = sin(pos / 10000^(256/512)) = sin(pos / 100)

        # High dimensions: High frequency (short wavelength, fast oscillation)
        PE(pos, 510) = sin(pos / 10000^(510/512)) ≈ sin(pos / 10000)
        ```

        **Intuition:**
        - **Low dimensions**: Capture "rough" position (beginning/middle/end)
        - **High dimensions**: Capture "precise" position (exact token index)

        Like a multi-resolution clock where different dimensions tick at different speeds!
        """)

with st.expander("Question 5: Modern developments"):
    st.markdown("""
    **Question**: Why have modern LLMs (LLaMA, Mistral, etc.) moved away from sinusoidal/learned positional encoding?

    A) They use RoPE which better handles relative positions and long contexts
    B) They don't use any positional encoding
    C) Sinusoidal encoding was too slow
    D) Learned embeddings used too much memory
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: A) They use RoPE which better handles relative positions and long contexts**

        **Explanation**:

        **Evolution of positional encoding:**

        **2017 - Original Transformer**: Sinusoidal encoding
        - Fixed mathematical function
        - Absolute positions

        **2018-2020 - BERT/GPT-2/GPT-3**: Learned embeddings
        - Adapts to data
        - Fixed max length

        **2021-2025 - Modern LLMs**: RoPE (Rotary Position Embeddings)
        - Used in: LLaMA, GPT-Neo, PaLM, Mistral, Gemini
        - Better relative position encoding
        - Excellent extrapolation to longer sequences
        - Works well for long-context tasks (32k+ tokens)

        **Why RoPE is better:**
        - Encodes relative positions more naturally
        - Better performance on long documents
        - Maintains performance when extending context length

        We'll learn RoPE in Topic 17!
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand how transformers learn word order through positional encoding!

But the story doesn't end here. Researchers discovered even better ways to encode position:

Next topics:
- **Topic 17**: RoPE (Rotary Position Embeddings) - the modern standard used in LLaMA, Mistral, and most new LLMs
- **Topic 18**: Complete Transformer Architecture - putting all the pieces together
- **Topic 19**: Modern Transformer Components - how current LLMs differ from the original

**You now understand why transformers can tell "dog bites man" from "man bites dog"!** 🚀

---

*Navigation: ← Previous: Topic 15 (Multi-Head Attention) | Next: Topic 17 (RoPE) →*
""")
