"""
Topic 11: Advanced Optimizers & Schedulers - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="11 - Advanced Optimizers & Schedulers",
    page_icon="📈",
    layout="wide"
)

# Main content
st.markdown("""
# Advanced Optimizers & Schedulers 📈

## Review: Basic Optimizers

We've already seen basic optimizers, but let's review:

### Stochastic Gradient Descent (SGD)

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**How it works**: Update weights by moving in the direction of negative gradient
```
w = w - lr * gradient
```

**Pros**: Simple, well-understood, works for most problems
**Cons**: Slow convergence, sensitive to learning rate, can get stuck in saddles

### Adam (Adaptive Moment Estimation)

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

**How it works**: Adapts learning rate per parameter using 1st and 2nd moments
```
m = β₁ * m + (1 - β₁) * gradient        # First moment (mean)
v = β₂ * v + (1 - β₂) * gradient²       # Second moment (variance)
w = w - lr * m / (√v + ε)                # Update with adapted LR
```

**Pros**: Fast convergence, adaptive learning rates, robust default hyperparameters
**Cons**: Can generalize slightly worse than SGD, uses more memory

---

## AdamW: Adam with Weight Decay

**AdamW** is Adam with proper weight decay (regularization):

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01    # L2 regularization
)
```

### What's the Difference?

**Adam with weight decay** (incorrect):
```python
# Weight decay applied BEFORE adaptive LR
gradient = gradient + weight_decay * w
# Then adaptive LR applied
```

**AdamW** (correct):
```python
# Adaptive LR applied first
update = adaptive_lr(gradient)
# Then weight decay applied directly to weights
w = w - lr * update - weight_decay * w
```

**Why AdamW is better**: Proper weight decay improves generalization!

### When to Use AdamW:

✅ **Default choice** for most deep learning tasks
✅ **Training transformers** (BERT, GPT, etc.)
✅ **When you need regularization** but don't want to hurt adaptive LR
✅ **Fine-tuning pre-trained models**

```python
# Modern best practice for transformers
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,              # Typical for fine-tuning
    betas=(0.9, 0.999),
    weight_decay=0.01     # L2 regularization
)
```

---

## RMSprop: Root Mean Square Propagation

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

**How it works**: Like Adam but only uses second moment (no first moment)

**Use cases**:
- Recurrent Neural Networks (RNNs, LSTMs)
- Online learning / streaming data
- Non-stationary problems

**Pros**: Good for RNNs, simpler than Adam
**Cons**: Less popular than Adam, fewer use cases

---

## Comparing Optimizers

| Optimizer | Speed | Generalization | Memory | Best For |
|-----------|-------|----------------|--------|----------|
| SGD | ⭐⭐ | ⭐⭐⭐⭐ | Low | CNNs, simple tasks |
| SGD + Momentum | ⭐⭐⭐ | ⭐⭐⭐⭐ | Low | CNNs, proven recipes |
| Adam | ⭐⭐⭐⭐ | ⭐⭐⭐ | High | Most tasks, fast prototyping |
| AdamW | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | High | Transformers, modern default |
| RMSprop | ⭐⭐⭐ | ⭐⭐⭐ | Medium | RNNs, LSTMs |

### Decision Guide:

```
Training transformers (BERT, GPT)?
└─> Use AdamW

Training CNNs for competition?
└─> Use SGD with momentum (better final accuracy)

Prototyping quickly?
└─> Use Adam (fast convergence)

Training RNNs?
└─> Use RMSprop or AdamW

Unsure?
└─> Start with AdamW, it works for most things!
```

---

## What are Learning Rate Schedulers?

**Problem**: A fixed learning rate is suboptimal

```
Fixed LR = 0.01:
Epoch 1-10:   Fast progress ✅
Epoch 10-50:  Still learning ✅
Epoch 50-100: Bouncing around, not converging ❌
```

**Solution**: Decrease learning rate during training!

### Why Schedulers Help:

1. **Early training**: Large LR for fast progress
2. **Mid training**: Medium LR for steady improvement
3. **Late training**: Small LR to fine-tune and converge

```
Loss landscape:

Large LR:  ●------>●  (jump over minimum)
Small LR:  ●->●->●    (converge to minimum)
```

---

## StepLR: Drop Every N Epochs

Reduce LR by a factor every N epochs:

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(
    optimizer,
    step_size=30,    # Reduce LR every 30 epochs
    gamma=0.1        # Multiply LR by 0.1
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)

    scheduler.step()  # Update learning rate

    # Check current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr}")

# LR schedule:
# Epoch 0-29:   LR = 0.1
# Epoch 30-59:  LR = 0.01   (×0.1)
# Epoch 60-89:  LR = 0.001  (×0.1)
# Epoch 90-99:  LR = 0.0001 (×0.1)
```

**Use case**: Simple baseline, works well for CNNs

---

## ExponentialLR: Smooth Decay

Multiply LR by gamma every epoch:

```python
from torch.optim.lr_scheduler import ExponentialLR

optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = ExponentialLR(
    optimizer,
    gamma=0.95    # Multiply LR by 0.95 every epoch
)

# LR schedule:
# Epoch 0:  LR = 0.100
# Epoch 1:  LR = 0.095  (×0.95)
# Epoch 2:  LR = 0.090  (×0.95)
# Epoch 3:  LR = 0.086  (×0.95)
# ...smooth decay
```

**Use case**: Smooth continuous decay, RNNs

---

## CosineAnnealingLR: Cosine Curve Decay

LR follows a cosine curve:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,      # Total epochs
    eta_min=1e-6    # Minimum LR
)

# LR follows cosine curve from 0.001 to 1e-6 over 100 epochs
```

**Visualization**:
```
LR
 │
 │  ╱‾‾‾╲
 │ ╱     ╲
 │╱       ╲___
 └─────────────> Epoch
```

**Why cosine?** Smooth decay with slow start and gentle landing!

**Use case**: **Most popular for transformers** (GPT, BERT, etc.)

### With Warmup (Recommended):

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Don't create scheduler yet - we'll implement warmup manually

def get_lr(epoch, warmup_epochs=10, total_epochs=100, max_lr=0.001, min_lr=1e-6):
    if epoch < warmup_epochs:
        # Linear warmup
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

# Training loop
import math
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(100):
    # Set LR manually
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train_one_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch}: LR = {lr:.6f}")

# LR schedule:
# Epoch 0-9:   Linear warmup from 0 to 0.001
# Epoch 10-99: Cosine decay from 0.001 to 1e-6
```

---

## ReduceLROnPlateau: Reduce When Stuck

Reduce LR when validation loss stops improving:

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',         # Minimize metric (loss)
    factor=0.1,         # Multiply LR by 0.1
    patience=10,        # Wait 10 epochs before reducing
    verbose=True        # Print when LR changes
)

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # ✅ Pass validation loss to scheduler!
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

# Example output:
# Epoch 20: Val loss hasn't improved for 10 epochs, reducing LR to 0.0001
```

**Use case**: When you don't know how many epochs to train, or for unpredictable tasks

**Pros**: Adaptive to your specific training run
**Cons**: Can reduce LR too early or too late

---

## Learning Rate Warmup

**Warmup** gradually increases LR from 0 to target over initial epochs:

### Why Warmup?

**Problem**: Large initial LR with random weights can destabilize training

```
No warmup:
Epoch 0: LR=0.001, loss=100 → loss=2.5  ✅
Epoch 1: LR=0.001, loss=2.5 → loss=0.8  ✅
Epoch 2: LR=0.001, loss=0.8 → loss=NaN  ❌ Exploded!

With warmup:
Epoch 0: LR=0.0001, loss=100 → loss=50    ✅
Epoch 1: LR=0.0002, loss=50 → loss=25     ✅
...stable training...
```

### Implementing Warmup:

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.max_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.current_epoch += 1

# Usage
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
warmup = WarmupScheduler(optimizer, warmup_epochs=5, max_lr=1e-3)

for epoch in range(100):
    warmup.step()  # Update LR for first 5 epochs
    train_one_epoch(model, train_loader, optimizer)
```

**Best practice for transformers**: 10-20% of total epochs for warmup

---

## Combining Warmup + Cosine Decay (Modern Best Practice)

This is the standard for training transformers:

```python
import math
import torch.optim as optim

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    \"\"\"
    Linear warmup + cosine decay scheduler
    Used by BERT, GPT, and most transformers
    \"\"\"
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

# Usage
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,      # Warmup for 1k steps
    num_training_steps=10000    # Total 10k steps
)

# Training loop (step-based, not epoch-based!)
for step, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # ✅ Step every batch, not every epoch!

    if step >= 10000:
        break
```

**Why step-based?** Modern transformers train for fixed number of steps, not epochs!

---

## Practical Example: Complete Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Model, data, etc.
model = MyModel()
train_loader = DataLoader(...)
val_loader = DataLoader(...)

# Optimizer and scheduler
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Training loop
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Update learning rate
    scheduler.step()

    # Log metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"  LR: {current_lr:.6f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("  ✅ Saved new best model!")

print("Training complete!")
```

---

## Connection to LLM Training

Modern LLM training uses sophisticated schedules:

### GPT-3 Style (Warmup + Cosine):

```python
# 375M tokens, batch size 0.5M tokens
# LR: Linear warmup for 375M tokens (750 steps)
# Then cosine decay to 10% of peak LR
optimizer = optim.AdamW(params, lr=6e-4, weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=300_000, eta_min=6e-5)
```

### LLaMA Style (AdamW + Cosine):

```python
# AdamW with β₁=0.9, β₂=0.95
# Cosine decay to 10% of peak LR
# Weight decay 0.1
optimizer = optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

**Key patterns**:
1. AdamW is the standard optimizer
2. Warmup for 2-10% of training
3. Cosine decay to 10% of peak LR
4. Weight decay 0.1 (strong regularization)

---

## Best Practices

### 1. Start with These Defaults

```python
# For CNNs (computer vision)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# For Transformers (NLP, modern architectures)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# For quick prototyping
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# No scheduler needed for initial experiments
```

### 2. Monitor Learning Rate

```python
# Log LR every epoch
for epoch in range(num_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")

    # Or use tensorboard
    writer.add_scalar('lr', current_lr, epoch)
```

### 3. Tune Learning Rate

```python
# If loss is NaN or exploding:
lr = lr / 10  # Reduce by 10×

# If loss isn't decreasing:
lr = lr * 2   # Increase by 2×

# If training is slow:
# Use warmup + larger LR + scheduler
```

### 4. Different LRs for Different Layers

```python
# Transfer learning: smaller LR for pre-trained layers
optimizer = optim.AdamW([
    {'params': model.base.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

---

## Key Takeaways 💡

✅ **AdamW**: Modern default optimizer (Adam + proper weight decay)
✅ **Cosine decay**: Most popular LR schedule for transformers
✅ **Warmup**: Stabilizes early training, use 10-20% of total epochs
✅ **ReduceLROnPlateau**: Adaptive schedule based on validation loss
✅ **Step-based scheduling**: Modern transformers use steps, not epochs
✅ **Best practices**: AdamW + warmup + cosine decay for transformers
✅ **Monitoring**: Always log current LR to understand training dynamics
✅ **Connection**: Same techniques used for GPT, BERT, LLaMA training!

**Next topic**: Learn about Model Saving and Checkpointing - how to save your training progress and deploy models!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What is the key difference between Adam and AdamW?",
        "options": [
            "AdamW is faster",
            "AdamW uses less memory",
            "AdamW applies weight decay correctly (after adaptive LR, not before)",
            "AdamW uses different momentum values"
        ],
        "correct": "AdamW applies weight decay correctly (after adaptive LR, not before)",
        "explanation": "AdamW applies weight decay directly to weights AFTER computing the adaptive learning rate update, while Adam with weight_decay applies it BEFORE (as part of gradient). This proper decoupling improves generalization and is why AdamW is the modern default for transformers."
    },
    {
        "question": "Why do we use learning rate warmup?",
        "options": [
            "To make training faster",
            "To gradually increase LR from 0 to target, preventing instability from large initial LR with random weights",
            "To reduce overfitting",
            "To save memory"
        ],
        "correct": "To gradually increase LR from 0 to target, preventing instability from large initial LR with random weights",
        "explanation": "With random initialization, a large initial learning rate can cause gradients to explode and training to diverge. Warmup gradually increases LR over the first few epochs (typically 10-20% of training), allowing the model to stabilize before full-speed training. This is critical for transformers!"
    },
    {
        "question": "What learning rate schedule is most commonly used for training transformers (BERT, GPT)?",
        "options": [
            "Constant LR (no schedule)",
            "StepLR (drop every N epochs)",
            "Linear warmup followed by cosine decay",
            "Exponential decay"
        ],
        "correct": "Linear warmup followed by cosine decay",
        "explanation": "Modern transformers (GPT, BERT, LLaMA) use linear warmup for 2-10% of training, then cosine decay to ~10% of peak LR. This gives stable early training (warmup) and smooth convergence (cosine decay). It's become the standard for NLP and increasingly for vision too!"
    },
    {
        "question": "When should you use ReduceLROnPlateau scheduler?",
        "options": [
            "Always, for every model",
            "When you want to reduce LR based on validation loss plateauing",
            "Only for CNNs",
            "Never, it's deprecated"
        ],
        "correct": "When you want to reduce LR based on validation loss plateauing",
        "explanation": "ReduceLROnPlateau monitors validation loss and reduces LR when it stops improving for 'patience' epochs. Use when you don't know ideal training length or for unpredictable tasks. Unlike fixed schedules (cosine, step), it adapts to your specific training run. Pass val_loss to scheduler.step(val_loss)!"
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
st.info("👈 Use the sidebar to navigate to the next topic: **12 - Model Saving and Checkpointing**")
