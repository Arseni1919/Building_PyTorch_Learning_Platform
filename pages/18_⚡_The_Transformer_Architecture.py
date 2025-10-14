import streamlit as st

st.title("⚡ Topic 18: The Transformer Architecture")

st.markdown("""
---

## 🎯 Putting It All Together: The Complete Picture

You've learned the building blocks:
- ✅ **Attention mechanism** (Topic 14): How to focus on relevant parts
- ✅ **Multi-head attention** (Topic 15): Multiple parallel attention patterns
- ✅ **Positional encoding** (Topic 16): Adding position information
- ✅ **RoPE** (Topic 17): Modern position encoding

**Now it's time to assemble them into THE architecture that changed AI forever!**

**The Transformer** - introduced in the 2017 paper "Attention is All You Need" - is the foundation of:
- 🤖 ChatGPT (GPT-3, GPT-4)
- 🤖 Claude (1, 2, 3, 4)
- 🤖 LLaMA (1, 2, 3)
- 🤖 BERT, T5, PaLM, Gemini, Mistral
- 🤖 Every major LLM you've heard of!

Let's build a complete transformer from scratch and understand how it works!

---

## 🏗️ The Original Transformer: Encoder-Decoder Architecture

The original transformer (2017) was designed for **translation** (English → German) and had two parts:

```
Input Sentence (English)
        ↓
    ENCODER (6 layers)
        ↓
    Context Representation
        ↓
    DECODER (6 layers)
        ↓
Output Sentence (German)
```

### Encoder: Understanding the Input

```
Input: "The cat sat on the mat"
  ↓
Embedding + Positional Encoding
  ↓
┌─────────────────────┐
│  Encoder Layer 1    │ → Self-Attention + Feed-Forward
├─────────────────────┤
│  Encoder Layer 2    │ → Self-Attention + Feed-Forward
├─────────────────────┤
│       ...           │
├─────────────────────┤
│  Encoder Layer 6    │ → Self-Attention + Feed-Forward
└─────────────────────┘
  ↓
Encoded Representation (understands the input)
```

**Purpose**: Build rich representations of the input sentence.

### Decoder: Generating the Output

```
Previously Generated: "Le chat"
  ↓
Embedding + Positional Encoding
  ↓
┌─────────────────────┐
│  Decoder Layer 1    │ → Self-Attention + Cross-Attention + Feed-Forward
├─────────────────────┤
│  Decoder Layer 2    │ → Self-Attention + Cross-Attention + Feed-Forward
├─────────────────────┤
│       ...           │
├─────────────────────┤
│  Decoder Layer 6    │ → Self-Attention + Cross-Attention + Feed-Forward
└─────────────────────┘
  ↓
Next Word: "assis"
```

**Purpose**: Generate output one word at a time, attending to both previously generated words AND the encoded input.

---

## 🔍 Modern Variants: Encoder-Only vs Decoder-Only

The original encoder-decoder is powerful, but modern LLMs use simplified versions:

### Encoder-Only: BERT

**Use case**: Understanding (classification, question answering, sentiment analysis)

```
Input: "The movie was [MASK]"
  ↓
Encoder (12 or 24 layers)
  ↓
Output: Prediction for [MASK] → "great"
```

**Examples**: BERT, RoBERTa, DeBERTa
**Key feature**: Bidirectional (can see future context)

### Decoder-Only: GPT

**Use case**: Generation (text completion, chatbots, creative writing)

```
Input: "Once upon a time"
  ↓
Decoder (96+ layers for GPT-3)
  ↓
Output: "there was a princess"
```

**Examples**: GPT-2, GPT-3, GPT-4, LLaMA, Claude, Mistral
**Key feature**: Causal (can only see past context)

**Why decoder-only dominates?**
- Simpler architecture (no encoder-decoder complexity)
- Scales incredibly well
- Excellent for few-shot learning
- Can be fine-tuned for any task

---

## 🧩 Transformer Block Components

Let's break down what's inside each transformer layer!

### Encoder Block

```python
Input: x [batch, seq_len, d_model]

┌────────────────────────────────────┐
│  1. Multi-Head Self-Attention      │
│     - Look at all input positions  │
│     - Attend to relevant words     │
└────────────────────────────────────┘
           ↓
       Add & Norm (residual connection + layer norm)
           ↓
┌────────────────────────────────────┐
│  2. Feed-Forward Network (MLP)     │
│     - Position-wise transformation │
│     - Add non-linearity            │
└────────────────────────────────────┘
           ↓
       Add & Norm (residual connection + layer norm)
           ↓
Output: x [batch, seq_len, d_model]
```

### Decoder Block

```python
Input: x [batch, seq_len, d_model]

┌────────────────────────────────────┐
│  1. Masked Multi-Head Attention    │
│     - Only attend to past tokens   │
│     - Causal masking               │
└────────────────────────────────────┘
           ↓
       Add & Norm
           ↓
┌────────────────────────────────────┐
│  2. Cross-Attention (if encoder)   │
│     - Attend to encoder output     │
│     - (Skipped in decoder-only)    │
└────────────────────────────────────┘
           ↓
       Add & Norm
           ↓
┌────────────────────────────────────┐
│  3. Feed-Forward Network (MLP)     │
│     - Position-wise transformation │
└────────────────────────────────────┘
           ↓
       Add & Norm
           ↓
Output: x [batch, seq_len, d_model]
```

---

## 💻 Complete PyTorch Implementation

Let's build a full transformer from scratch!

### 1. Feed-Forward Network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension (e.g., 512)
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = self.linear1(x)        # [batch, seq_len, d_ff]
        x = F.relu(x)              # ReLU activation
        x = self.dropout(x)
        x = self.linear2(x)        # [batch, seq_len, d_model]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (from Topic 15)"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, V)
        output = self.combine_heads(attention_output)
        output = self.W_o(output)

        return output


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer

    Components:
    1. Multi-Head Self-Attention
    2. Add & Norm (residual + layer norm)
    3. Feed-Forward Network
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]

        # 1. Self-Attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 2. Feed-Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Components:
    1. Masked Multi-Head Self-Attention (causal)
    2. Add & Norm
    3. Cross-Attention (optional, for encoder-decoder)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output=None, src_mask=None, tgt_mask=None):
        # x: [batch, seq_len, d_model]

        # 1. Masked Self-Attention (causal)
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # 2. Cross-Attention (if encoder-decoder architecture)
        if encoder_output is not None:
            cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))

        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder)

    This is the architecture from "Attention is All You Need"!
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final linear layer
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_seq_len, d_model):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, max_seq_len, d_model]

    def encode(self, src, src_mask=None):
        # src: [batch, src_seq_len]

        # Embedding + Positional Encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = src_emb + self.pos_encoding[:, :src.size(1), :].to(src.device)
        src_emb = self.dropout(src_emb)

        # Pass through encoder layers
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)

        return encoder_output

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # tgt: [batch, tgt_seq_len]

        # Embedding + Positional Encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
        tgt_emb = self.dropout(tgt_emb)

        # Pass through decoder layers
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)

        return decoder_output

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Final linear projection
        output = self.output_linear(decoder_output)

        return output


# Example: Create a transformer
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers
)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 🔑 Critical Components Explained

### 1. Residual Connections (Add)

```python
# Without residual:
x = layer(x)  # Information can get lost through deep networks

# With residual:
x = x + layer(x)  # Original information preserved!
```

**Why essential?**
- Enables training very deep networks (100+ layers)
- Gradients flow directly backward
- Prevents vanishing gradients

### 2. Layer Normalization (Norm)

```python
x = LayerNorm(x + layer(x))
```

**Why needed?**
- Stabilizes training
- Reduces internal covariate shift
- Allows higher learning rates

**How it works:**
```python
# Normalize across d_model dimension
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_normalized = (x - mean) / (std + epsilon)
```

### 3. Feed-Forward Network

```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Purpose:**
- Add non-linearity and capacity
- Process each position independently
- Typically d_ff = 4 × d_model (expands then compresses)

**Why after attention?**
- Attention gathers information
- FFN processes that information

### 4. Causal Masking (for Decoders)

```python
# Create lower triangular mask
mask = torch.tril(torch.ones(seq_len, seq_len))

"""
[1 0 0 0]  Position 0 can only see position 0
[1 1 0 0]  Position 1 can see positions 0-1
[1 1 1 0]  Position 2 can see positions 0-2
[1 1 1 1]  Position 3 can see positions 0-3
"""
```

**Why?**
- Prevents "cheating" by seeing future tokens
- Essential for autoregressive generation
- Used in GPT, LLaMA, Claude

---

## 🎯 GPT Architecture (Decoder-Only)

Modern LLMs like ChatGPT use a simplified decoder-only architecture:

```python
class GPTBlock(nn.Module):
    """
    GPT-style Transformer Block (Decoder-only)

    Simpler than encoder-decoder - just masked self-attention + FFN!
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture (modern variant)
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    """
    GPT-style Transformer (like ChatGPT architecture)
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        # idx: [batch, seq_len]
        batch_size, seq_len = idx.shape

        # Embeddings
        token_emb = self.token_embedding(idx)  # [batch, seq_len, d_model]
        pos_emb = self.position_embedding(torch.arange(seq_len, device=idx.device))
        x = self.dropout(token_emb + pos_emb)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)  # [batch, seq_len, vocab_size]

        return logits


# Example: GPT-2 Small configuration
gpt_model = GPT(
    vocab_size=50257,      # GPT-2 vocabulary
    d_model=768,           # Embedding dimension
    num_heads=12,          # 12 attention heads
    num_layers=12,         # 12 transformer layers
    d_ff=3072,             # FFN hidden dimension (4 * 768)
    max_seq_len=1024,      # Maximum context length
    dropout=0.1
)

print(f"GPT-2 Small parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")
# Output: ~117M parameters (matches GPT-2 Small!)
```

---

## 📊 Architecture Configurations: Real LLMs

### GPT-2 (OpenAI)

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 762M |
| XL | 48 | 1600 | 25 | 1.5B |

### GPT-3 (OpenAI)

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| Small | 12 | 768 | 12 | 125M |
| Medium | 24 | 1024 | 16 | 350M |
| Large | 24 | 1536 | 16 | 760M |
| XL | 24 | 2048 | 24 | 1.3B |
| 2.7B | 32 | 2560 | 32 | 2.7B |
| 6.7B | 32 | 4096 | 32 | 6.7B |
| 13B | 40 | 5140 | 40 | 13B |
| **175B** | **96** | **12288** | **96** | **175B** |

### LLaMA (Meta)

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| 7B | 32 | 4096 | 32 | 7B |
| 13B | 40 | 5120 | 40 | 13B |
| 33B | 60 | 6656 | 52 | 33B |
| 65B | 80 | 8192 | 64 | 65B |

**Pattern:**
- Larger models: More layers + wider (larger d_model)
- d_ff typically = 4 × d_model
- num_heads scales with d_model

---

## 🔄 Training a Transformer

### Causal Language Modeling (GPT-style)

```python
# Objective: Predict next token
def train_step(model, batch):
    input_ids = batch[:, :-1]   # "The cat sat"
    targets = batch[:, 1:]      # "cat sat on"

    # Forward pass
    logits = model(input_ids)   # [batch, seq_len, vocab_size]

    # Compute loss
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss
```

### Masked Language Modeling (BERT-style)

```python
# Objective: Predict masked tokens
def train_step(model, batch):
    input_ids = batch.clone()

    # Randomly mask 15% of tokens
    mask_indices = torch.rand(input_ids.shape) < 0.15
    input_ids[mask_indices] = MASK_TOKEN_ID

    # Forward pass
    logits = model(input_ids)

    # Compute loss only on masked positions
    loss = F.cross_entropy(
        logits[mask_indices],
        batch[mask_indices]
    )

    loss.backward()
    optimizer.step()

    return loss
```

---

## 🚀 From Transformer to Modern LLMs

### Original Transformer (2017)
- Encoder-decoder
- Sinusoidal positional encoding
- LayerNorm
- ReLU activation

### Modern LLMs (2024-2025)
- Decoder-only (simpler, scales better)
- RoPE (better position encoding)
- RMSNorm (faster normalization)
- SwiGLU (better activation)
- GQA (memory-efficient attention)
- Flash Attention (faster computation)

**We'll cover these modern improvements in Topics 19-23!**

---

## 🎓 Key Takeaways

1. **Transformer = Attention + FFN + Residuals + Norms**, stacked in layers
2. **Three variants**: Encoder-decoder (translation), Encoder-only (BERT), Decoder-only (GPT)
3. **Encoder**: Bidirectional self-attention for understanding
4. **Decoder**: Causal self-attention for generation
5. **Residual connections**: Enable deep networks (100+ layers)
6. **Layer normalization**: Stabilize training
7. **Feed-forward networks**: Add capacity and non-linearity
8. **GPT is decoder-only**: Simpler but extremely powerful when scaled
9. **Modern LLMs are all transformers**: Variations on this core architecture

**The transformer architecture is THE foundation of all modern LLMs!**

---

## 📝 Quiz Time!

Test your understanding of the complete transformer architecture.
""")

# Quiz questions
with st.expander("Question 1: Encoder vs Decoder"):
    st.markdown("""
    **Question**: What is the key difference between transformer encoder and decoder?

    A) Encoder is deeper than decoder
    B) Encoder uses bidirectional attention, decoder uses causal (masked) attention
    C) Encoder doesn't use attention
    D) Decoder doesn't use feed-forward networks
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) Encoder uses bidirectional attention, decoder uses causal (masked) attention**

        **Explanation**:

        **Encoder (BERT-style):**
        ```python
        # Can attend to ALL positions (past and future)
        mask = None  # or all 1s
        attention(x, x, x, mask=mask)
        ```
        Use case: Understanding tasks (classification, QA)

        **Decoder (GPT-style):**
        ```python
        # Can only attend to PAST positions (causal masking)
        mask = torch.tril(...)  # Lower triangular
        attention(x, x, x, mask=mask)
        ```
        Use case: Generation tasks (text completion, chatbots)

        The causal masking in decoders prevents "cheating" by seeing future tokens during generation!
        """)

with st.expander("Question 2: Why residual connections?"):
    st.markdown("""
    **Question**: Why are residual connections (x = x + layer(x)) essential in transformers?

    A) They make the model faster
    B) They enable training very deep networks by preventing vanishing gradients
    C) They reduce memory usage
    D) They make the model more interpretable
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: B) They enable training very deep networks by preventing vanishing gradients**

        **Explanation**:

        **Without residuals:**
        ```python
        x = layer1(x)
        x = layer2(x)
        ...
        x = layer100(x)  # Gradients vanish! Training fails!
        ```

        **With residuals:**
        ```python
        x = x + layer1(x)  # Gradient highway!
        x = x + layer2(x)
        ...
        x = x + layer100(x)  # Gradients flow directly back! ✓
        ```

        **Benefits:**
        - GPT-3: 96 layers
        - GPT-4: Likely 100+ layers
        - LLaMA: 80 layers

        Without residual connections, training such deep networks would be impossible!

        **Bonus**: Residuals also preserve the original information, preventing information loss.
        """)

with st.expander("Question 3: Feed-forward network purpose"):
    st.markdown("""
    **Question**: What is the purpose of the Feed-Forward Network (FFN) in each transformer layer?

    A) To add positional information
    B) To add non-linearity and processing capacity after attention gathers information
    C) To normalize the activations
    D) To reduce the dimensionality
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) To add non-linearity and processing capacity after attention gathers information**

        **Explanation**:

        **Attention**: Gathers relevant information
        ```python
        x = self_attention(x)  # "What information is relevant?"
        ```

        **FFN**: Processes that information
        ```python
        x = FFN(x)  # "What should I do with this information?"
        ```

        **FFN structure:**
        ```python
        FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
        # Expand: d_model → d_ff (typically 4× larger)
        # Apply ReLU non-linearity
        # Compress: d_ff → d_model
        ```

        **Why needed:**
        - Attention is linear transformation (matrix multiplication)
        - FFN adds non-linearity (ReLU/GELU)
        - More parameters = more capacity
        - Processing happens independently for each position

        Typically 2/3 of transformer parameters are in FFN layers!
        """)

with st.expander("Question 4: GPT vs BERT architecture"):
    st.markdown("""
    **Question**: Why do modern LLMs (GPT-4, Claude, LLaMA) use decoder-only architecture instead of encoder-decoder?

    A) Decoder-only is faster to train
    B) Decoder-only is simpler and scales better while being versatile for many tasks
    C) Decoder-only uses less memory
    D) Encoder-decoder doesn't work well
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) Decoder-only is simpler and scales better while being versatile for many tasks**

        **Explanation**:

        **Encoder-Decoder (Original Transformer, T5):**
        - More complex (two separate stacks)
        - Good for specific tasks (translation, summarization)
        - Harder to scale to 100B+ parameters

        **Decoder-Only (GPT, LLaMA, Claude):**
        - Simpler architecture
        - Scales incredibly well (GPT-3: 175B, GPT-4: likely 1T+)
        - Versatile: Can do ANY task with prompting
        - Emergent abilities at scale (few-shot learning)
        - Easier to train at massive scale

        **The shift:**
        - 2017-2019: Encoder-decoder popular
        - 2020+: Decoder-only dominates

        **Why decoder-only won:**
        - Simplicity enables scaling
        - Scaling unlocks emergent abilities
        - Prompting works better than expected
        - Can fine-tune for specific tasks if needed

        Modern insight: **Scale + simplicity > architectural complexity**
        """)

with st.expander("Question 5: Layer stacking"):
    st.markdown("""
    **Question**: What happens when we stack more transformer layers (e.g., 12 → 96 layers)?

    A) The model becomes faster
    B) The model can learn more complex and hierarchical representations
    C) The model uses less memory
    D) The model requires less training data
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: B) The model can learn more complex and hierarchical representations**

        **Explanation**:

        **Hierarchical processing in deep transformers:**

        ```
        Layer 1-10:   Surface patterns (syntax, grammar)
        Layer 11-30:  Semantic relationships (meaning)
        Layer 31-60:  Complex reasoning (logic, inference)
        Layer 61-96:  Abstract concepts (world knowledge, common sense)
        ```

        **Research findings:**
        - Early layers: Learn syntax ("is" follows "he")
        - Middle layers: Learn semantics (cat = animal)
        - Late layers: Learn reasoning (if X then Y)

        **Real examples:**
        - GPT-2 Small (12 layers): Basic completion
        - GPT-3 (96 layers): Few-shot learning, reasoning
        - GPT-4 (100+ layers): Advanced reasoning, multimodal

        **The pattern:**
        More layers = More abstraction = More capable

        But: Diminishing returns + harder to train + more compute
        """)

st.markdown("""
---

## 🎯 What's Next?

You now understand the complete transformer architecture - the foundation of ALL modern LLMs!

But modern LLMs don't use the exact 2017 architecture. They've evolved with better components:

Next topics:
- **Topic 19**: Modern Transformer Components (RMSNorm, SwiGLU) - what makes LLaMA different from original transformer
- **Topic 20**: Grouped Query Attention (GQA) - memory-efficient attention for large models
- **Topic 21**: Flash Attention - 2-4x faster attention computation
- **Topic 22**: KV Cache - efficient autoregressive generation

**You now understand the architecture behind ChatGPT, Claude, LLaMA, and every modern LLM!** 🚀

---

*Navigation: ← Previous: Topic 17 (RoPE) | Next: Topic 19 (Modern Transformer Components) →*
""")
