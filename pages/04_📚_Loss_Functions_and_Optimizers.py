"""
Topic 4: Loss Functions & Optimizers - Basic Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="04 - Loss Functions & Optimizers",
    page_icon="🎯",
    layout="wide"
)

# Main content
st.markdown("""
# Loss Functions & Optimizers 🎯

## The Learning Loop

Training a neural network has three key steps:

1. **Forward pass**: Make predictions
2. **Calculate loss**: Measure how wrong the predictions are
3. **Backward pass + Update**: Calculate gradients and adjust weights

Loss functions and optimizers handle steps 2 and 3!

---

## What is a Loss Function?

A loss function (or cost function) **measures how bad your model's predictions are**. It's a single number that answers: "How wrong am I?"

**The goal of training**: Minimize the loss!

### Common Loss Functions

#### 1. CrossEntropyLoss (Classification)

**Use case**: Multi-class classification (MNIST, ImageNet, etc.)

```python
import torch
import torch.nn as nn

# Example: 3 samples, 5 classes
predictions = torch.randn(3, 5)  # Raw logits (unnormalized scores)
targets = torch.tensor([0, 2, 1])  # True class indices

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(predictions, targets)
print(f"Loss: {loss.item():.4f}")
```

**Key points**:
- Expects **raw logits** (no softmax needed!)
- Combines softmax + negative log-likelihood internally
- Lower is better (perfect predictions = 0 loss)

**Why it matters**: This is the standard loss for LLM training (predicting next token)!

#### 2. MSELoss (Regression)

**Use case**: Predicting continuous values (house prices, temperatures, etc.)

```python
predictions = torch.tensor([2.5, 0.0, 2.1])
targets = torch.tensor([3.0, -0.5, 2.0])

loss_fn = nn.MSELoss()
loss = loss_fn(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")  # Mean of squared differences
```

Formula: `MSE = (1/n) * Σ(predicted - actual)²`

#### 3. BCELoss (Binary Classification)

**Use case**: Yes/no decisions (spam detection, sentiment analysis)

```python
predictions = torch.sigmoid(torch.randn(4))  # Probabilities between 0-1
targets = torch.tensor([1., 0., 1., 0.])

loss_fn = nn.BCELoss()
loss = loss_fn(predictions, targets)
```

**Important**: BCELoss expects probabilities (apply sigmoid first!). Or use `BCEWithLogitsLoss` which applies sigmoid internally.

---

## What is an Optimizer?

An optimizer **updates the model's weights** to reduce the loss. It implements the gradient descent algorithm.

Think of it as hiking down a mountain (the loss landscape) to reach the valley (minimum loss).

### Common Optimizers

#### 1. SGD (Stochastic Gradient Descent)

The simplest optimizer - just follow the gradient downhill:

```python
import torch.optim as optim

model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training step
optimizer.zero_grad()        # Clear old gradients
loss.backward()              # Calculate new gradients
optimizer.step()             # Update weights
```

**Parameters**:
- `lr` (learning rate): Step size (too high = overshoot, too low = slow training)

**With momentum** (helps escape local minima):
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### 2. Adam (Adaptive Moment Estimation)

The most popular optimizer - adapts learning rate per parameter:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Why it's better**:
- Adjusts learning rate automatically for each parameter
- Works well with default settings
- Faster convergence on most problems

**Default hyperparameters**:
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,           # Learning rate
    betas=(0.9, 0.999), # Exponential decay rates
    eps=1e-8            # Numerical stability
)
```

#### 3. AdamW (Adam with Weight Decay)

Adam with better regularization (prevents overfitting):

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Used by**: Modern transformers (GPT, BERT, LLaMA, etc.)

---

## Complete Training Step

Here's how loss functions and optimizers work together:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Setup
model = MyNeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for one batch
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

    # Backward pass + optimization
    optimizer.zero_grad()  # ① Clear old gradients
    loss.backward()        # ② Compute new gradients
    optimizer.step()       # ③ Update weights

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

**Critical order**:
1. ✅ `zero_grad()` BEFORE `backward()`
2. ✅ `backward()` BEFORE `step()`

---

## Why zero_grad() is Essential

Gradients **accumulate** by default in PyTorch. If you don't clear them, old gradients add to new ones!

```python
# ❌ WRONG - gradients accumulate!
loss.backward()
optimizer.step()

# ✅ CORRECT - clear old gradients first
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**When accumulation is useful**: Gradient accumulation for large batch sizes (advanced technique).

---

## Learning Rate: The Most Important Hyperparameter

The learning rate controls how big each weight update is:

```python
# Too high (lr=1.0) - might overshoot and diverge
optimizer = optim.Adam(model.parameters(), lr=1.0)  # ❌

# Too low (lr=0.0001) - training will be very slow
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 🐌

# Just right (lr=0.001) - common default for Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ✅
```

**Rules of thumb**:
- Adam: Start with `lr=0.001` (or `1e-3`)
- SGD: Start with `lr=0.01` (needs higher values)
- Use learning rate schedulers to decrease lr over time (covered in Intermediate level)

---

## Loss Function Cheat Sheet

| Task | Loss Function | Target Format |
|------|---------------|---------------|
| Multi-class classification | `nn.CrossEntropyLoss()` | Class indices [0, 1, 2, ...] |
| Binary classification | `nn.BCEWithLogitsLoss()` | Binary labels [0, 1] |
| Regression | `nn.MSELoss()` | Continuous values |
| Multi-label classification | `nn.BCEWithLogitsLoss()` | Binary vectors [0, 1, 1, 0, ...] |

---

## Optimizer Comparison

| Optimizer | Speed | Memory | When to Use |
|-----------|-------|--------|-------------|
| SGD | Fast | Low | Simple problems, fine-tuning |
| SGD + Momentum | Fast | Low | Better than plain SGD |
| Adam | Medium | High | Default choice for most tasks |
| AdamW | Medium | High | Modern transformers, prevents overfitting |

---

## Connection to Transformers

Modern LLMs use:
- **Loss**: CrossEntropyLoss (predict next token)
- **Optimizer**: AdamW with weight decay
- **Learning rate**: Small (1e-4 to 1e-5) with warmup + decay schedule

Example from GPT training:
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=6e-4,           # 0.0006
    betas=(0.9, 0.95), # Slightly different from defaults
    weight_decay=0.1   # Strong regularization
)
```

---

## Key Takeaways 💡

✅ **Loss functions** measure prediction error (lower is better)
✅ **Optimizers** update weights using gradients to minimize loss
✅ Always call `optimizer.zero_grad()` before `loss.backward()`
✅ Adam is the default choice for most problems
✅ Learning rate is the most important hyperparameter to tune
✅ Modern transformers use AdamW + CrossEntropyLoss

**Next step**: Put everything together by training your first model!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What does CrossEntropyLoss expect as input?",
        "options": [
            "Probabilities after softmax",
            "Raw logits (unnormalized scores)",
            "Binary values (0 or 1)",
            "One-hot encoded vectors"
        ],
        "correct": "Raw logits (unnormalized scores)",
        "explanation": "CrossEntropyLoss expects raw logits and applies softmax internally for numerical stability. If you apply softmax yourself, the loss calculation will be incorrect."
    },
    {
        "question": "What is the correct order of operations in the training loop?",
        "options": [
            "forward → backward → zero_grad → step",
            "forward → loss → zero_grad → backward → step",
            "zero_grad → forward → loss → step → backward",
            "backward → zero_grad → forward → loss → step"
        ],
        "correct": "forward → loss → zero_grad → backward → step",
        "explanation": "The correct order is: (1) forward pass to get predictions, (2) calculate loss, (3) zero_grad to clear old gradients, (4) backward to compute new gradients, (5) step to update weights."
    },
    {
        "question": "Which optimizer is most commonly used for training modern transformers?",
        "options": [
            "SGD",
            "SGD with momentum",
            "Adam",
            "AdamW"
        ],
        "correct": "AdamW",
        "explanation": "AdamW (Adam with Weight Decay) is the standard optimizer for modern transformers like GPT, BERT, and LLaMA. It provides better regularization than plain Adam."
    },
    {
        "question": "Which loss function should you use for predicting house prices?",
        "options": [
            "CrossEntropyLoss",
            "MSELoss",
            "BCELoss",
            "NLLLoss"
        ],
        "correct": "MSELoss",
        "explanation": "House price prediction is a regression task (predicting continuous values), so MSELoss (Mean Squared Error) is appropriate. CrossEntropyLoss and BCELoss are for classification tasks."
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
st.info("👈 Use the sidebar to navigate to the next topic: **05 - Training Your First Model**")
