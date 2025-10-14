import streamlit as st

st.title("⚡ Topic 14: Attention Mechanism from Scratch")

st.markdown("""
---

## 🎯 The Revolution That Changed Everything

Welcome to **the most important concept in modern AI**. Everything you've heard about - ChatGPT, Claude, GPT-4, BERT, LLaMA - they all rely on one fundamental idea: **Attention**.

In 2017, a paper titled "Attention is All You Need" revolutionized artificial intelligence. Today, we'll build attention from scratch and understand why it's so powerful.

---

## 🤔 The Problem: Why RNNs Failed

Before attention, we used **Recurrent Neural Networks (RNNs)** for sequences like text:

```python
# RNNs process sequences one step at a time
word1 → RNN → hidden1
word2 → RNN → hidden2 (depends on hidden1)
word3 → RNN → hidden3 (depends on hidden2)
```

**Problems with RNNs:**

1. **Sequential Processing**: Can't parallelize - must wait for previous word
2. **Vanishing Gradients**: Information from early words gets lost in long sequences
3. **Fixed Context**: Everything compressed into one hidden state vector
4. **No Direct Connections**: Word 1 can't directly influence word 100

**Example of the problem:**
```
Input: "The cat, which was sitting on the mat that was placed in the corner, was sleeping."
Question: What was sleeping?
Answer: The cat (mentioned 15 words ago!)
```

RNNs struggle because "The cat" information has to travel through 15 steps to reach "sleeping".

---

## 💡 The Breakthrough: Attention

**Core Idea**: Instead of forcing information through a sequential bottleneck, let every word **directly look at every other word** and decide what's relevant.

**Attention in one sentence:**
> "For each word, look at all other words and focus on the ones that matter most."

**Why it's revolutionary:**
- ✅ **Parallel Processing**: All words processed simultaneously
- ✅ **Direct Connections**: Any word can directly attend to any other word
- ✅ **No Information Loss**: No compression through hidden states
- ✅ **Interpretable**: Can visualize which words attend to which

---

## 🧩 The Query, Key, Value Intuition

Attention uses three concepts: **Query**, **Key**, and **Value**. Let's build intuition:

### 📚 Library Analogy

Imagine you're in a library searching for information:

- **Query (Q)**: "What am I looking for?" (Your search question)
- **Key (K)**: "What am I about?" (Book titles/topics on shelves)
- **Value (V)**: "What information do I contain?" (Book contents)

**Process:**
1. You have a Query: "machine learning algorithms"
2. You scan all book Keys (titles) and find which ones match your query
3. You retrieve the Values (contents) from the most relevant books
4. You combine information from multiple relevant books

### 🔤 Text Example

Sentence: "The cat sat on the mat"

When processing the word "sat":
- **Query**: "What am I (sat) related to?"
- **Keys**: ["The", "cat", "sat", "on", "the", "mat"] - all words
- **Values**: The actual representations of those words

The attention mechanism:
1. Compares "sat" query with all keys
2. Finds "cat" and "mat" are most relevant
3. Retrieves their values
4. Creates a weighted combination

---

## 🔢 Scaled Dot-Product Attention: The Math

Don't worry - the math is simpler than it looks!

### The Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Let's break it down **step by step**:

### Step 1: Compute Attention Scores

```python
scores = Q @ K.T  # Dot product: How similar is each query to each key?
```

**Why dot product?**
- High dot product = vectors point in similar directions = related concepts
- Low dot product = vectors point in different directions = unrelated

**Example:**
```
Q: [0.9, 0.1] (represents "cat")
K1: [0.8, 0.2] (represents "animal") → high dot product ✓
K2: [0.1, 0.9] (represents "number") → low dot product ✗
```

### Step 2: Scale by √d_k

```python
d_k = Q.shape[-1]  # Dimension of key vectors
scores = scores / math.sqrt(d_k)
```

**Why scale?**
- Without scaling, dot products get very large with high dimensions
- Large numbers → softmax saturation → tiny gradients → poor training
- √d_k normalizes scores to reasonable range

### Step 3: Apply Softmax

```python
attention_weights = softmax(scores)
```

**Why softmax?**
- Converts scores to probabilities (sum to 1)
- Emphasizes differences (high scores get higher, low scores get lower)
- Creates a weighted distribution

### Step 4: Weighted Sum of Values

```python
output = attention_weights @ V
```

**Why multiply by V?**
- We don't just want to know what's relevant (weights)
- We want to retrieve the actual information (values)
- Weighted sum = "mix the relevant information"

---

## 💻 Complete PyTorch Implementation

Let's build attention from scratch!

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from 'Attention is All You Need'

    This is the CORE mechanism behind all modern LLMs!
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: Query matrix [batch_size, seq_len, d_k]
            K: Key matrix [batch_size, seq_len, d_k]
            V: Value matrix [batch_size, seq_len, d_v]
            mask: Optional mask [batch_size, seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, d_v]
            attention_weights: Attention scores [batch_size, seq_len, seq_len]
        """
        # Step 1: Compute attention scores
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]

        # Step 2: Scale by √d_k
        scores = scores / math.sqrt(d_k)

        # Step 3: Apply mask (optional, for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Step 4: Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


# Example usage
batch_size = 2
seq_len = 4
d_model = 8

# Random query, key, value matrices
Q = torch.randn(batch_size, seq_len, d_model)
K = torch.randn(batch_size, seq_len, d_model)
V = torch.randn(batch_size, seq_len, d_model)

# Create attention module
attention = ScaledDotProductAttention()

# Apply attention
output, weights = attention(Q, K, V)

print(f"Input Q shape: {Q.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\\nAttention weights (first sample):\\n{weights[0]}")
```

**Output:**
```
Input Q shape: torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
Attention weights shape: torch.Size([2, 4, 4])

Attention weights (first sample):
tensor([[0.2123, 0.2841, 0.2156, 0.2880],
        [0.3012, 0.1876, 0.2543, 0.2569],
        [0.2234, 0.2987, 0.2145, 0.2634],
        [0.2567, 0.2234, 0.2678, 0.2521]])
```

---

## 👁️ Visualizing Attention Weights

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(attention_weights, tokens=None):
    """
    Visualize attention weights as a heatmap
    """
    # Get first sample
    weights = attention_weights[0].detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap='viridis', aspect='auto')

    # Labels
    if tokens is None:
        tokens = [f"Token {i}" for i in range(len(weights))]

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    ax.set_xlabel("Keys (attending to)")
    ax.set_ylabel("Queries (attending from)")
    ax.set_title("Attention Weights Heatmap")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig

# Example with actual tokens
tokens = ["The", "cat", "sat", "mat"]
fig = visualize_attention(weights, tokens)
```

**Interpretation:**
- Bright colors = high attention (these words are related)
- Dark colors = low attention (these words are not related)
- Each row shows what one word attends to

---

## 🔄 Self-Attention vs Cross-Attention

### Self-Attention
**Used in**: GPT, BERT, LLaMA, Claude

Q, K, V all come from the **same sequence**:
```python
# Self-attention: sentence attends to itself
sentence = "The cat sat on the mat"
Q = K = V = embed(sentence)
output = attention(Q, K, V)
```

**Purpose**: Understanding relationships within one sequence

**Examples:**
- "The cat" → attends to → "sat" (subject-verb relationship)
- "it" → attends to → "cat" (pronoun resolution)

### Cross-Attention
**Used in**: Translation models, image captioning, CLIP

Q comes from one sequence, K and V from another:
```python
# Cross-attention: translation example
english = "The cat sat"
french = "Le chat"

Q = embed(french)      # What I'm translating
K = V = embed(english)  # What I'm translating from
output = attention(Q, K, V)
```

**Purpose**: Connecting two different sequences

---

## 🚀 Why Attention Revolutionized NLP

### Before Attention (2016)
- RNNs, LSTMs struggling with long sequences
- Translation quality plateauing
- Sequential processing = slow training

### After Attention (2017-2025)
- ✅ **Transformers**: Pure attention, no recurrence
- ✅ **BERT (2018)**: Pre-trained language understanding
- ✅ **GPT-2 (2019)**: Large-scale text generation
- ✅ **GPT-3 (2020)**: Few-shot learning
- ✅ **ChatGPT (2022)**: Conversational AI
- ✅ **GPT-4, Claude 3 (2023-2024)**: Multimodal reasoning
- ✅ **LLaMA 3, Gemini (2024-2025)**: Open source revolution

**Every single modern LLM uses attention as its core mechanism!**

---

## 🌟 Connection to Modern LLMs

### GPT Architecture
```
Input tokens
  ↓
Self-Attention (what we just learned!)
  ↓
Feed-Forward Network
  ↓
(Repeat 96 times for GPT-3)
  ↓
Output tokens
```

### What Makes Different LLMs Unique?
They all use attention, but differ in:
- **Number of layers**: GPT-3 has 96, LLaMA 3 has 80
- **Attention type**: Multi-head (next topic!), Grouped Query Attention
- **Position encoding**: Sinusoidal → RoPE (Topic 17)
- **Normalization**: LayerNorm → RMSNorm (Topic 19)
- **Efficiency**: Flash Attention (Topic 21), KV Cache (Topic 22)

But **scaled dot-product attention** is the foundation of them all!

---

## 🎓 Key Takeaways

1. **Attention solves RNN limitations**: Direct connections, parallelization, no information loss
2. **Query-Key-Value intuition**: Search mechanism for relevant information
3. **Scaled dot-product formula**: Simple but powerful - 4 steps
4. **Self-attention**: Sequence attends to itself (GPT, BERT)
5. **Cross-attention**: One sequence attends to another (translation)
6. **Foundation of modern AI**: Every LLM since 2017 uses attention

**This is THE breakthrough that enabled ChatGPT, Claude, and all modern AI!**

---

## 📝 Quiz Time!

Test your understanding of attention mechanisms.
""")

# Quiz questions
with st.expander("Question 1: What is the main advantage of attention over RNNs?"):
    st.markdown("""
    **Question**: What is the PRIMARY advantage of attention over RNNs for long sequences?

    A) Attention has fewer parameters
    B) Attention allows parallel processing and direct connections between all positions
    C) Attention trains faster on small datasets
    D) Attention uses less memory
    """)

    if st.button("Show Answer", key="q1"):
        st.success("""
        **Answer: B) Attention allows parallel processing and direct connections between all positions**

        **Explanation**: The key advantage is that attention creates direct connections between any two positions in a sequence, regardless of distance. RNNs must propagate information sequentially, causing:
        - Vanishing gradients over long distances
        - Sequential processing (can't parallelize)
        - Information bottleneck through hidden states

        Attention solves all three problems, which is why it revolutionized NLP!
        """)

with st.expander("Question 2: What do Query, Key, and Value represent?"):
    st.markdown("""
    **Question**: In the attention mechanism, what role does the Key (K) play?

    A) It contains the information to be retrieved
    B) It represents what each position is looking for
    C) It represents what each position is "about" for matching
    D) It scales the attention scores
    """)

    if st.button("Show Answer", key="q2"):
        st.success("""
        **Answer: C) It represents what each position is "about" for matching**

        **Explanation**:
        - **Query (Q)**: "What am I looking for?" - the search query
        - **Key (K)**: "What am I about?" - used to match against queries
        - **Value (V)**: "What information do I contain?" - the actual content retrieved

        The Query-Key dot product determines relevance, then we retrieve the corresponding Values.
        """)

with st.expander("Question 3: Why do we scale by √d_k?"):
    st.markdown("""
    **Question**: Why do we divide attention scores by √d_k?

    A) To make computation faster
    B) To prevent large dot products that cause softmax saturation
    C) To add regularization
    D) To match the dimensions of Q and K
    """)

    if st.button("Show Answer", key="q3"):
        st.success("""
        **Answer: B) To prevent large dot products that cause softmax saturation**

        **Explanation**: When d_k is large (e.g., 512), dot products can become very large. Without scaling:

        ```
        scores = [50, 45, 2, 1]  (large values)
        softmax = [0.993, 0.007, 0.0, 0.0]  (saturated!)
        ```

        With scaling by √d_k:
        ```
        scores = [2.2, 2.0, 0.09, 0.04]  (moderate values)
        softmax = [0.45, 0.40, 0.08, 0.07]  (good distribution!)
        ```

        This prevents vanishing gradients during training.
        """)

with st.expander("Question 4: Self-Attention vs Cross-Attention"):
    st.markdown("""
    **Question**: What is the difference between self-attention and cross-attention?

    A) Self-attention uses one head, cross-attention uses multiple heads
    B) Self-attention has Q=K=V from same sequence, cross-attention has Q from one sequence and K=V from another
    C) Self-attention is used in GPT, cross-attention is used in BERT
    D) Self-attention is bidirectional, cross-attention is unidirectional
    """)

    if st.button("Show Answer", key="q4"):
        st.success("""
        **Answer: B) Self-attention has Q=K=V from same sequence, cross-attention has Q from one sequence and K=V from another**

        **Explanation**:

        **Self-Attention**: A sequence attends to itself
        - Used in: GPT, BERT, LLaMA, Claude
        - Q = K = V = same input sequence
        - Purpose: Understanding relationships within one sequence

        **Cross-Attention**: One sequence attends to another
        - Used in: Translation models, encoder-decoder architectures
        - Q from one sequence, K=V from another
        - Purpose: Connecting two different sequences (e.g., English → French)
        """)

with st.expander("Question 5: Real-World Impact"):
    st.markdown("""
    **Question**: Which of the following modern AI systems does NOT use attention as its core mechanism?

    A) ChatGPT (GPT-4)
    B) Claude 3
    C) Traditional Decision Trees
    D) BERT
    """)

    if st.button("Show Answer", key="q5"):
        st.success("""
        **Answer: C) Traditional Decision Trees**

        **Explanation**:

        **Systems that use attention:**
        - ChatGPT/GPT-4: Transformer decoder with self-attention
        - Claude 3: Transformer-based architecture
        - BERT: Transformer encoder with self-attention
        - LLaMA, Mistral, Gemini: All use attention
        - DALL-E, Stable Diffusion: Use attention for images
        - Whisper: Uses attention for speech recognition

        **Systems that DON'T use attention:**
        - Traditional ML: Decision Trees, Random Forests, SVMs
        - Classical NLP: Rule-based systems

        Since 2017, attention has become THE foundational mechanism for all modern deep learning in NLP, vision, and multimodal AI!
        """)

st.markdown("""
---

## 🎯 What's Next?

You've learned the core attention mechanism - the foundation of modern AI!

In the next topics, we'll build on this to understand:
- **Topic 15**: Multi-Head Attention (why GPT uses 96+ attention heads)
- **Topic 16**: Positional Encoding (teaching transformers about word order)
- **Topic 17**: RoPE (the modern alternative used in LLaMA)
- **Topic 18**: Complete Transformer architecture

**You now understand the secret behind ChatGPT, Claude, and every modern LLM!** 🚀

---

*Navigation: ← Previous: Topic 13 (Embeddings) | Next: Topic 15 (Multi-Head Attention) →*
""")
