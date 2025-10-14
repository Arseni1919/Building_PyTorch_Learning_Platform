"""
Topic 9: Batch Normalization & Dropout - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="09 - Batch Normalization & Dropout",
    page_icon="🎯",
    layout="wide"
)

# Main content
st.markdown("""
# Batch Normalization & Dropout 🎯

## What is Batch Normalization?

**Batch Normalization (BatchNorm)** is a technique that normalizes layer inputs during training.

### The Problem It Solves: Internal Covariate Shift

As training progresses, layer inputs change distribution:

```
Initial distribution:      After 100 iterations:    After 1000 iterations:
    ┌───┐                      ┌───┐                    ┌───┐
    │   │                      │   │                    │   │
────┼───┼────           ───────┼───┼────         ──────────┼───┼────
mean=0, std=1           mean=5, std=10          mean=50, std=100
```

**Why this is bad**:
- Later layers must constantly adapt to changing inputs
- Training becomes unstable
- Need very small learning rates
- Deeper networks become nearly impossible to train

**BatchNorm solution**: Normalize inputs to each layer, making training more stable!

---

## How Batch Normalization Works

BatchNorm has 4 steps per mini-batch:

### Step 1: Compute Mean and Variance

For each feature in the batch:
```
μ = (1/m) Σ x_i        # Mean across batch
σ² = (1/m) Σ (x_i - μ)²   # Variance across batch
```

### Step 2: Normalize

```
x̂ = (x - μ) / √(σ² + ε)
```
ε is a small constant (10⁻⁵) to avoid division by zero.

### Step 3: Scale and Shift (Learnable!)

```
y = γ * x̂ + β
```

Where:
- **γ (gamma)**: Learnable scale parameter
- **β (beta)**: Learnable shift parameter

**Why learnable?** Allows the network to undo normalization if needed!

### Step 4: Update Running Statistics (for inference)

During training, track exponential moving average:
```
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

During inference, use these running statistics instead of batch statistics!

---

## Batch Normalization in PyTorch

### For Fully-Connected Layers:

```python
import torch
import torch.nn as nn

class NetworkWithBN(nn.Module):
    def __init__(self):
        super(NetworkWithBN, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Normalize 256 features

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Normalize 128 features

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # Linear -> BatchNorm -> ReLU
        x = self.fc1(x)
        x = self.bn1(x)     # Normalize before activation
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)     # Normalize before activation
        x = torch.relu(x)

        x = self.fc3(x)
        return x

model = NetworkWithBN()
print(model)
```

### For Convolutional Layers:

```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super(CNNWithBN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Normalize 64 channels

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Normalize 128 channels

        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 56 * 56, 10)

    def forward(self, x):
        # Conv -> BatchNorm -> ReLU -> Pool
        x = self.conv1(x)
        x = self.bn1(x)      # Normalize feature maps
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)      # Normalize feature maps
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNNWithBN()
```

### Key BatchNorm Variants:

| Layer Type | BatchNorm | Input Shape | Use Case |
|-----------|-----------|-------------|----------|
| `nn.Linear` | `BatchNorm1d(features)` | `[batch, features]` | Fully-connected |
| `nn.Conv2d` | `BatchNorm2d(channels)` | `[batch, channels, H, W]` | Convolutional |
| `nn.Conv3d` | `BatchNorm3d(channels)` | `[batch, channels, D, H, W]` | 3D Conv |

---

## When to Use BatchNorm

### Standard Placement:

```python
# Option 1: Conv/Linear -> BatchNorm -> Activation (Most Common)
x = self.conv1(x)
x = self.bn1(x)
x = torch.relu(x)

# Option 2: Conv/Linear -> Activation -> BatchNorm (Less Common)
x = self.conv1(x)
x = torch.relu(x)
x = self.bn1(x)
```

**Recommendation**: Use Option 1 (BatchNorm before activation). This is the standard practice!

### Don't Use BatchNorm:

❌ **On the final layer** (output layer)
```python
# BAD - Don't normalize final predictions!
x = self.fc_final(x)
x = self.bn_final(x)  # ❌ No!
return x
```

❌ **With very small batch sizes** (batch_size < 4)
- Statistics become unstable with tiny batches
- Consider GroupNorm or LayerNorm instead

---

## What is Dropout?

**Dropout** randomly sets neurons to zero during training to prevent overfitting.

### The Problem It Solves: Overfitting

Models can memorize training data instead of learning general patterns:

```
Training accuracy: 99%  ✅
Test accuracy: 60%      ❌ Overfitting!
```

**Why overfitting happens**:
- Too many parameters
- Too few training samples
- Network learns spurious correlations

### How Dropout Works:

During **training**, randomly zero out neurons with probability p:

```
Before Dropout:           After Dropout (p=0.5):
┌───┬───┬───┬───┐        ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │   -->  │ 0 │ 2 │ 0 │ 4 │  (50% dropped)
└───┴───┴───┴───┘        └───┴───┴───┴───┘

Then scale remaining:     ┌───┬───┬───┬───┐
                          │ 0 │ 4 │ 0 │ 8 │  (×2 to keep sum)
                          └───┴───┴───┴───┘
```

During **inference**, use all neurons (no dropout):
```
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │  All active!
└───┴───┴───┴───┘
```

**Why this prevents overfitting**:
- Forces network to learn redundant representations
- Prevents neurons from co-adapting (relying on specific neurons)
- Effectively trains an ensemble of networks

---

## Dropout in PyTorch

### Basic Usage:

```python
import torch
import torch.nn as nn

class NetworkWithDropout(nn.Module):
    def __init__(self):
        super(NetworkWithDropout, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(p=0.5)  # Drop 50% of neurons

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.5)  # Drop 50% of neurons

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)    # Apply dropout after activation

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)    # Apply dropout after activation

        x = self.fc3(x)
        return x

model = NetworkWithDropout()
```

### Training vs Evaluation Mode:

**CRITICAL**: You must switch modes!

```python
# Training mode - dropout is ACTIVE
model.train()
output = model(x)  # Some neurons randomly zeroed

# Evaluation mode - dropout is DISABLED
model.eval()
output = model(x)  # All neurons active
```

### What Happens Internally:

```python
# Training mode
model.train()
x = torch.tensor([1., 2., 3., 4.])
dropout = nn.Dropout(p=0.5)
print(dropout(x))  # tensor([0., 4., 0., 8.])  Random!

# Evaluation mode
model.eval()
print(dropout(x))  # tensor([1., 2., 3., 4.])  No change!
```

---

## Where to Place Dropout

### For Fully-Connected Networks:

```python
class FCNetwork(nn.Module):
    def __init__(self):
        super(FCNetwork, self).__init__()

        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Pattern: Linear -> Activation -> Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)        # ✅ After activation

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)        # ✅ After activation

        x = self.fc3(x)             # ✅ No dropout on output!
        return x
```

### For CNNs:

```python
class CNNWithDropout(nn.Module):
    def __init__(self):
        super(CNNWithDropout, self).__init__()

        # Convolutional layers (usually NO dropout here)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully-connected layers (dropout HERE)
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.dropout = nn.Dropout(0.5)  # Only in FC layers!
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Conv layers - no dropout
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers - with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)         # ✅ Dropout in FC layers
        x = self.fc2(x)
        return x
```

**Rule of thumb**: Dropout is most useful in fully-connected layers, less so in convolutional layers!

---

## Common Dropout Rates

| Layer Type | Dropout Rate | Reasoning |
|-----------|--------------|-----------|
| Input layer | 0.1 - 0.2 | Low rate, preserve input info |
| Hidden FC layers | 0.5 | Standard rate for regularization |
| Output layer | 0.0 | Never dropout predictions! |
| Conv layers | 0.0 - 0.2 | Usually lower or none |

### Tuning Dropout:

```python
# Start with 0.5 for FC layers
dropout_rate = 0.5

# If overfitting persists, increase
dropout_rate = 0.6  # More aggressive

# If underfitting (train loss high), decrease
dropout_rate = 0.3  # Less aggressive

# Monitor train vs validation accuracy
```

---

## BatchNorm vs Dropout: When to Use Which?

### Batch Normalization:

**Use when:**
- Training deep networks (helps stability)
- Want faster convergence
- Have reasonable batch sizes (≥16)
- Want to use higher learning rates

**Benefits:**
- Stabilizes training
- Allows higher learning rates
- Reduces sensitivity to initialization
- Slight regularization effect

### Dropout:

**Use when:**
- Model is overfitting (train acc >> val acc)
- Have many parameters
- Training fully-connected layers
- Dataset is small

**Benefits:**
- Strong regularization
- Prevents co-adaptation
- Ensemble effect

### Can You Use Both?

**Yes!** They complement each other:

```python
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()

        # Conv layers with BatchNorm
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)      # BatchNorm for stability

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)     # BatchNorm for stability

        self.pool = nn.MaxPool2d(2, 2)

        # FC layers with BatchNorm AND Dropout
        self.fc1 = nn.Linear(128 * 56 * 56, 512)
        self.bn3 = nn.BatchNorm1d(512)     # BatchNorm for stability
        self.dropout = nn.Dropout(0.5)     # Dropout for regularization

        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Conv: Conv -> BN -> ReLU -> Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC: Linear -> BN -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x
```

**Pattern**: BatchNorm everywhere, Dropout in FC layers!

---

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class BestPracticeModel(nn.Module):
    def __init__(self, num_classes=10):
        super(BestPracticeModel, self).__init__()

        # Feature extraction with BatchNorm
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classification with BatchNorm and Dropout
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # ✅ Enable BatchNorm training mode and Dropout
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100. * correct / total

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()  # ✅ Disable BatchNorm training mode and Dropout
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradients for evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass only
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100. * correct / total

# Initialize
device = torch.device('cpu')
model = BestPracticeModel(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Model ready with BatchNorm and Dropout!")
```

---

## Key Takeaways 💡

✅ **BatchNorm**: Normalizes layer inputs, stabilizes training, allows higher learning rates
✅ **BatchNorm formula**: Normalize → Scale (γ) → Shift (β) [learnable!]
✅ **Place BatchNorm**: After linear/conv, before activation (standard practice)
✅ **Dropout**: Randomly zeros neurons during training to prevent overfitting
✅ **Dropout rate**: 0.5 for FC layers, lower for conv layers, 0 for output
✅ **Training vs Eval**: Must call `model.train()` or `model.eval()` - critical!
✅ **Use both**: BatchNorm for stability + Dropout for regularization
✅ **Pattern**: Conv/Linear → BatchNorm → ReLU → Dropout (in FC layers)

**Next topic**: Learn about Transfer Learning - leveraging pre-trained models to save time and get better results!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What problem does Batch Normalization solve?",
        "options": [
            "Overfitting due to too many parameters",
            "Internal covariate shift - changing input distributions during training",
            "Vanishing gradients in ReLU activations",
            "Slow inference speed"
        ],
        "correct": "Internal covariate shift - changing input distributions during training",
        "explanation": "BatchNorm addresses internal covariate shift, where layer inputs change distribution as training progresses. By normalizing inputs to each layer, training becomes more stable, allows higher learning rates, and makes deep networks easier to train."
    },
    {
        "question": "Why does Dropout prevent overfitting?",
        "options": [
            "It reduces the number of parameters in the model",
            "It forces neurons to learn redundant representations and prevents co-adaptation",
            "It normalizes the activations",
            "It speeds up training"
        ],
        "correct": "It forces neurons to learn redundant representations and prevents co-adaptation",
        "explanation": "Dropout randomly zeros neurons during training, forcing the network to learn redundant representations. Neurons can't rely on specific other neurons always being present, preventing co-adaptation. This acts like training an ensemble of networks, improving generalization."
    },
    {
        "question": "What happens if you forget to call model.eval() before testing with a model that has Dropout?",
        "options": [
            "Nothing - it works the same",
            "The model will train during testing",
            "Dropout will still randomly zero neurons, giving inconsistent/poor results",
            "The model will crash"
        ],
        "correct": "Dropout will still randomly zero neurons, giving inconsistent/poor results",
        "explanation": "Without model.eval(), Dropout remains active during testing and will randomly zero neurons. This gives inconsistent predictions (different results each run) and typically lower accuracy. Always call model.eval() for testing and model.train() for training!"
    },
    {
        "question": "Where should you typically place BatchNorm in a network?",
        "options": [
            "Only at the input layer",
            "Only at the output layer",
            "After linear/conv layers, before activation (standard practice)",
            "After the activation function"
        ],
        "correct": "After linear/conv layers, before activation (standard practice)",
        "explanation": "The standard placement is: Conv/Linear → BatchNorm → Activation. This normalizes the pre-activation values, which has become the widely accepted best practice. Don't use BatchNorm on the output layer."
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
st.info("👈 Use the sidebar to navigate to the next topic: **10 - Transfer Learning**")
