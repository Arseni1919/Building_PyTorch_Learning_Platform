"""
Topic 13: Introduction to Embeddings - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="13 - Introduction to Embeddings",
    page_icon="🔤",
    layout="wide"
)

# Main content
st.markdown("""
# Introduction to Embeddings 🔤

## What Are Embeddings?

**Embeddings** are learned representations that map discrete objects (words, items, users) into continuous vector spaces.

### The Problem: Representing Discrete Data

Neural networks work with continuous numbers (tensors), but many inputs are discrete:

```
Words: "cat", "dog", "computer"
User IDs: user_12345, user_67890
Movie IDs: movie_001, movie_002
Categories: "electronics", "books", "clothing"
```

**How do we feed these into neural networks?**

---

## Naive Approach: One-Hot Encoding

Represent each item as a binary vector:

```python
# Vocabulary: ["cat", "dog", "bird", "fish"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

### Problems with One-Hot Encoding:

❌ **Huge dimensionality**: 100,000 words = 100,000 dimensions!
❌ **No similarity**: "cat" and "dog" are equally distant from each other
❌ **No semantic meaning**: Can't capture that "cat" and "dog" are both animals
❌ **Memory inefficient**: Mostly zeros (sparse)

```python
# One-hot for vocabulary of 50,000 words
vocab_size = 50000
one_hot = torch.zeros(50000)  # 50k numbers, 49,999 are zero!
one_hot[word_id] = 1
```

---

## Solution: Embeddings (Dense Vectors)

Map each item to a **learned dense vector** of fixed size:

```python
# Vocabulary: ["cat", "dog", "bird", "fish"]
# Embedding dimension: 3

"cat"  → [0.2, -0.5, 0.9]   # 3 numbers, all meaningful!
"dog"  → [0.3, -0.4, 0.8]   # Similar to cat
"bird" → [-0.1, 0.7, 0.2]   # Different from cat/dog
"fish" → [-0.2, 0.6, -0.3]  # Different from others
```

### Benefits of Embeddings:

✅ **Low dimensionality**: 50,000 words → 300 dimensions (typical)
✅ **Captures similarity**: Similar items have similar vectors
✅ **Learns semantic meaning**: "king" - "man" + "woman" ≈ "queen"
✅ **Efficient**: Dense vectors, no wasted space
✅ **Learned from data**: Network discovers useful representations!

---

## PyTorch Embedding Layer

PyTorch provides `nn.Embedding` for creating embedding layers:

```python
import torch
import torch.nn as nn

# Create embedding layer
embedding = nn.Embedding(
    num_embeddings=10000,  # Vocabulary size (number of unique items)
    embedding_dim=300       # Embedding dimension (size of each vector)
)

# Input: tensor of indices
input_ids = torch.tensor([5, 42, 9, 1000])  # Batch of 4 words

# Forward pass: lookup embeddings
embedded = embedding(input_ids)

print(embedded.shape)  # torch.Size([4, 300])
# 4 words, each represented as a 300-dimensional vector
```

### What Happens Internally?

```python
# Embedding layer is a lookup table (matrix)
# Shape: [num_embeddings, embedding_dim]

embedding.weight.shape  # torch.Size([10000, 300])

# input_ids = [5, 42, 9]
# Output:
#   embedding.weight[5]   → [0.1, -0.2, ..., 0.5]  (300 numbers)
#   embedding.weight[42]  → [0.3, 0.1, ..., -0.2]  (300 numbers)
#   embedding.weight[9]   → [-0.4, 0.7, ..., 0.1]  (300 numbers)
```

**It's just matrix indexing!** But the embeddings are **learned** during training.

---

## Complete Embedding Example

### Simple Word Embedding Model:

```python
import torch
import torch.nn as nn

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()

        # Embedding layer: maps word IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Classifier: average embeddings, then classify
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_length]
        # Each element is a word ID (integer)

        # Embed words
        embedded = self.embedding(x)  # [batch, seq_length, embedding_dim]

        # Average embeddings across sequence
        pooled = embedded.mean(dim=1)  # [batch, embedding_dim]

        # Classify
        output = self.fc(pooled)  # [batch, num_classes]

        return output

# Example usage
vocab_size = 10000      # 10k unique words
embedding_dim = 300     # Each word → 300D vector
num_classes = 5         # 5 sentiment classes

model = SimpleTextClassifier(vocab_size, embedding_dim, num_classes)

# Input: batch of sentences (word IDs)
# sentence 1: [45, 123, 9, 1000, 3]  (5 words)
# sentence 2: [67, 234, 12, 5, 89]   (5 words)
input_ids = torch.tensor([
    [45, 123, 9, 1000, 3],
    [67, 234, 12, 5, 89]
])  # [batch=2, seq_length=5]

# Forward pass
output = model(input_ids)
print(output.shape)  # torch.Size([2, 5])
# 2 samples, 5 classes
```

---

## Choosing Embedding Dimensions

How many dimensions should embeddings have?

### Common Choices:

| Vocabulary Size | Embedding Dim | Use Case |
|-----------------|---------------|----------|
| < 1,000 | 50-100 | Small datasets, categories |
| 1,000-10,000 | 100-300 | Medium vocabulary |
| 10,000-100,000 | 300-512 | Large vocabulary (NLP) |
| > 100,000 | 512-1024 | Very large (BERT: 768, GPT: 1024+) |

### Rule of Thumb:

```python
embedding_dim = int(vocab_size ** 0.25)

# Examples:
# vocab_size = 10,000  → embedding_dim ≈ 100
# vocab_size = 100,000 → embedding_dim ≈ 178
```

**Trade-offs**:
- **Too small**: Can't capture enough information
- **Too large**: Overfitting, slow training, more memory
- **Just right**: Depends on data size and task complexity!

---

## Word Embeddings: Word2Vec and GloVe

Before transformers, word embeddings were trained separately:

### Word2Vec (Google, 2013)

Learns word embeddings by predicting context:

```
Input:  "The cat sat on the mat"

Task:   Given "sat", predict ["the", "cat", "on", "the"]
        (predict surrounding words)

Result: Words that appear in similar contexts get similar embeddings!
```

**Famous example**:
```
king - man + woman ≈ queen
paris - france + germany ≈ berlin
```

### GloVe (Stanford, 2014)

Learns embeddings from word co-occurrence statistics:

```
Count how often words appear together:
- "cat" and "animal" appear together often → similar embeddings
- "cat" and "physics" rarely together → different embeddings
```

### Using Pre-trained Embeddings:

```python
import torch
import torch.nn as nn

# Load pre-trained GloVe embeddings (hypothetical)
pretrained_embeddings = load_glove_embeddings()  # [vocab_size, 300]

# Create embedding layer with pre-trained weights
embedding = nn.Embedding(vocab_size, embedding_dim=300)
embedding.weight = nn.Parameter(pretrained_embeddings)

# Option 1: Freeze embeddings (don't train)
embedding.weight.requires_grad = False

# Option 2: Fine-tune embeddings (train with small LR)
embedding.weight.requires_grad = True
```

**Modern approach**: Train embeddings end-to-end with your model (no pre-training needed for most tasks!)

---

## Position Embeddings (Preview)

In transformers, we also embed **positions**:

### The Problem:

Transformers process all words simultaneously (no sequential order like RNNs):

```
Input:  [the, cat, sat]
Output: [h1, h2, h3]     # No positional information!

"the cat sat" and "sat cat the" look the same to the model!
```

### The Solution: Position Embeddings

Add learnable position embeddings to word embeddings:

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(TransformerEmbedding, self).__init__()

        # Word embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(max_length, embedding_dim)

    def forward(self, input_ids):
        # input_ids: [batch, seq_length]

        seq_length = input_ids.size(1)

        # Word embeddings
        word_embeds = self.word_embedding(input_ids)  # [batch, seq_length, dim]

        # Position embeddings
        positions = torch.arange(0, seq_length, device=input_ids.device)
        position_embeds = self.position_embedding(positions)  # [seq_length, dim]

        # Add them together!
        embeddings = word_embeds + position_embeds  # [batch, seq_length, dim]

        return embeddings

# Example
vocab_size = 10000
embedding_dim = 512
max_length = 512

model = TransformerEmbedding(vocab_size, embedding_dim, max_length)

input_ids = torch.tensor([[5, 42, 9, 1000]])  # [1, 4]
output = model(input_ids)
print(output.shape)  # torch.Size([1, 4, 512])
```

**Now the model knows**: word at position 0 is different from the same word at position 5!

We'll explore this more in Advanced topics (Positional Encoding, RoPE, etc.)!

---

## Embedding Layers are Trainable

Embeddings are learned during training via backpropagation:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model with embedding
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 2)  # Binary sentiment

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        return self.fc(pooled)

model = SentimentModel(vocab_size=10000, embedding_dim=100)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for input_ids, labels in train_loader:
    # Forward pass
    outputs = model(input_ids)
    loss = criterion(outputs, labels)

    # Backward pass - embeddings are updated!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After training, embeddings have learned meaningful representations!
# Similar words will have similar embeddings
```

---

## Visualizing Embeddings

Embeddings live in high-dimensional space, but we can visualize with dimensionality reduction:

### t-SNE Visualization (Hypothetical):

```python
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract learned embeddings
embedding_matrix = model.embedding.weight.data.cpu()  # [vocab_size, embedding_dim]

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embedding_matrix[:1000])  # First 1000 words

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Annotate some words
words = ['cat', 'dog', 'computer', 'phone']
for word in words:
    word_id = vocab[word]
    x, y = embeddings_2d[word_id]
    plt.annotate(word, (x, y))

plt.title('Word Embeddings Visualization')
plt.show()
```

**What you'll see**: Similar words cluster together!
- Animals: cat, dog, bird cluster
- Technology: computer, phone, internet cluster
- Emotions: happy, sad, angry cluster

---

## Beyond Word Embeddings

Embeddings are used for all types of discrete data:

### 1. User Embeddings (Recommendation Systems)

```python
# Map user IDs to vectors
user_embedding = nn.Embedding(num_users=100000, embedding_dim=128)

# Similar users have similar embeddings
user_123 → [0.2, -0.5, ...]  # Likes sci-fi movies
user_456 → [0.3, -0.4, ...]  # Also likes sci-fi (similar vector!)
```

### 2. Item Embeddings (Recommendation Systems)

```python
# Map product IDs to vectors
item_embedding = nn.Embedding(num_items=50000, embedding_dim=128)

# Recommendation: dot product of user and item embeddings
score = torch.dot(user_embedding(user_id), item_embedding(item_id))
```

### 3. Token Embeddings (Transformers)

```python
# GPT, BERT, etc.
token_embedding = nn.Embedding(vocab_size=50257, embedding_dim=1024)

# Each token (word/subword) gets an embedding
```

### 4. Category Embeddings (Tabular Data)

```python
# Embed categorical features in tabular data
category_embedding = nn.Embedding(num_categories=20, embedding_dim=8)

# Example: product category, user age group, etc.
```

---

## Connection to Transformers

Embeddings are the **first layer** in all transformer models:

```python
# Simplified transformer input layer
class TransformerInput(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerInput, self).__init__()

        # Token embeddings (what we learned today!)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Position embeddings (we'll learn more in Advanced topics)
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, input_ids):
        # Embed tokens
        token_embeds = self.token_embedding(input_ids)

        # Embed positions
        seq_length = input_ids.size(1)
        positions = torch.arange(seq_length, device=input_ids.device)
        position_embeds = self.position_embedding(positions)

        # Combine (usually by addition)
        return token_embeds + position_embeds

# This is the input to BERT, GPT, and all transformers!
```

**What's coming in Advanced topics**:
- Positional Encoding (sinusoidal)
- RoPE (Rotary Position Embeddings)
- Why transformers need position information
- Modern position encoding methods

---

## Practical Example: Sentiment Classification

Complete example using embeddings:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SentimentClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hidden layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [batch_size, seq_length]

        # Embed
        embedded = self.embedding(x)  # [batch, seq_length, embedding_dim]

        # Average pooling
        pooled = embedded.mean(dim=1)  # [batch, embedding_dim]

        # Classify
        hidden = torch.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)

        return output

# Create model
model = SentimentClassifier(
    vocab_size=10000,
    embedding_dim=300,
    hidden_dim=128,
    num_classes=2  # Positive/Negative
)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
model.train()
for epoch in range(10):
    for input_ids, labels in train_loader:
        # input_ids: [batch, seq_length] - sentences as word IDs
        # labels: [batch] - sentiment labels

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

print("Training complete! Embeddings learned semantic representations.")
```

---

## Key Takeaways 💡

✅ **Embeddings**: Map discrete items (words, IDs) to continuous vectors
✅ **Why embeddings?**: Capture similarity, semantic meaning, efficient representation
✅ **nn.Embedding**: PyTorch layer for creating embedding lookup tables
✅ **Learned**: Embeddings are trained via backpropagation
✅ **Dimension**: Typically 100-512 for words, depends on vocab size and task
✅ **Position embeddings**: Add positional information for transformers
✅ **Word2Vec/GloVe**: Pre-trained word embeddings (less common now)
✅ **Universal**: Used for words, users, items, categories, tokens, etc.
✅ **Foundation**: First layer in all transformer models (BERT, GPT, LLaMA)

**Congratulations!** You've completed the Intermediate level. Next up: Advanced topics focusing on **Transformers** - starting with the Attention Mechanism!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "Why are embeddings better than one-hot encoding for representing words?",
        "options": [
            "Embeddings are faster to compute",
            "Embeddings are dense, low-dimensional, capture similarity, and learn semantic meaning",
            "Embeddings use less memory only",
            "One-hot encoding is actually better"
        ],
        "correct": "Embeddings are dense, low-dimensional, capture similarity, and learn semantic meaning",
        "explanation": "One-hot encoding creates huge sparse vectors (50k words = 50k dimensions) with no semantic meaning ('cat' and 'dog' are equally distant). Embeddings map to dense low-dimensional vectors (50k words → 300 dims) where similar words have similar vectors, and they learn semantic relationships (king-man+woman≈queen)!"
    },
    {
        "question": "What does nn.Embedding(10000, 300) create?",
        "options": [
            "A lookup table with 10,000 rows and 300 columns",
            "A neural network with 10,000 neurons",
            "300 different embedding matrices",
            "A tokenizer for 10,000 words"
        ],
        "correct": "A lookup table with 10,000 rows and 300 columns",
        "explanation": "nn.Embedding(num_embeddings, embedding_dim) creates a learnable matrix of shape [10000, 300]. Each row is an embedding vector for one item. When you pass in index [5], it returns row 5 (a 300-dim vector). It's essentially a lookup table where embeddings are learned during training!"
    },
    {
        "question": "Why do transformers need position embeddings in addition to word embeddings?",
        "options": [
            "To increase model size",
            "Because transformers process all words simultaneously and don't inherently know word order",
            "To make training faster",
            "Position embeddings aren't actually needed"
        ],
        "correct": "Because transformers process all words simultaneously and don't inherently know word order",
        "explanation": "Unlike RNNs that process words sequentially, transformers process all words in parallel through self-attention. Without position embeddings, 'the cat sat' and 'sat cat the' would look identical! Position embeddings (learned or sinusoidal) add positional information so the model knows word order."
    },
    {
        "question": "How are embedding vectors learned during training?",
        "options": [
            "They are manually set by humans",
            "They are computed from word statistics",
            "They are updated via backpropagation like other neural network parameters",
            "They are downloaded from pre-trained models"
        ],
        "correct": "They are updated via backpropagation like other neural network parameters",
        "explanation": "Embedding layers contain learnable parameters (the embedding matrix). During training, gradients flow back to these embeddings, and they're updated by the optimizer (SGD, Adam, etc.) just like other weights. The model learns embeddings that are most useful for the task - similar words naturally get similar embeddings!"
    }
]

for idx, q in enumerate(questions):
    st.markdown(f"### Question {idx + 1}")
    st.markdown(f"**{q['question']}**")

    user_answer = st.radio(
        "Select your answer:",
        options=q["options"],
        key=f"q{idx}",
        index=None
    )

    if st.button(f"Check Answer {idx + 1}", key=f"btn{idx}"):
        if user_answer:
            if user_answer == q["correct"]:
                st.success(f"✅ Correct! {q['explanation']}")
            else:
                st.error(f"❌ Incorrect. {q['explanation']}")
        else:
            st.warning("Please select an answer first!")

    st.markdown("---")

# Navigation
st.info("👈 Use the sidebar to navigate to the Advanced section: **14 - Attention Mechanism from Scratch**")
