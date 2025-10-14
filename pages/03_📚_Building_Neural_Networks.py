"""
Topic 3: Building Neural Networks (nn.Module) - Basic Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="03 - Building Neural Networks",
    page_icon="🧠",
    layout="wide"
)

# Main content
st.markdown("""
# Building Neural Networks (nn.Module) 🧠

## Why Do We Need nn.Module?

When building neural networks, we need:
- **Organized structure** for layers and parameters
- **Automatic parameter tracking** (so optimizers know what to update)
- **Easy model saving/loading**
- **GPU transfer capabilities**
- **Built-in training/evaluation modes**

PyTorch's `nn.Module` provides ALL of these features! Think of it as a **blueprint** for building neural networks.

---

## The Anatomy of nn.Module

Every neural network in PyTorch inherits from `nn.Module` and has two key components:

### 1. `__init__()` - Define Your Layers

This is where you **declare** all your layers (but don't connect them yet):

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()  # ALWAYS call this first!

        # Define layers
        self.fc1 = nn.Linear(784, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)   # Hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer
```

**Why `super().__init__()`?** It initializes the parent class so PyTorch can track your parameters automatically.

### 2. `forward()` - Connect Your Layers

This is where you **define the data flow** through your network:

```python
    def forward(self, x):
        # Data flows through layers
        x = torch.relu(self.fc1(x))  # Apply fc1, then ReLU
        x = torch.relu(self.fc2(x))  # Apply fc2, then ReLU
        x = self.fc3(x)              # Final output (no activation)
        return x
```

**Why no activation on the last layer?** The loss function (like CrossEntropyLoss) applies it internally!

---

## Complete Neural Network Example

Let's build a classifier for MNIST digits:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()

        # Layer definitions
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flatten image (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(x.size(0), -1)

        # Forward pass
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Randomly drop 20% of neurons
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model instance
model = MNISTClassifier()
print(model)
```

Output:
```
MNISTClassifier(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

---

## Making Predictions

```python
# Create dummy input (batch of 4 images)
dummy_input = torch.randn(4, 28, 28)

# Forward pass
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # [4, 10] - 4 samples, 10 classes

# Get predicted classes
predictions = torch.argmax(output, dim=1)
print(f"Predictions: {predictions}")  # [7, 2, 1, 5] - predicted digits
```

**Key insight**: When you call `model(x)`, PyTorch automatically calls `model.forward(x)`.

---

## Accessing Model Parameters

```python
# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # 109,386

# View specific layer parameters
print(f"fc1 weight shape: {model.fc1.weight.shape}")  # [128, 784]
print(f"fc1 bias shape: {model.fc1.bias.shape}")      # [128]

# Iterate through named parameters (useful for debugging)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

---

## Training vs Evaluation Mode

Neural networks behave differently during training vs inference:

```python
# Training mode (default) - dropout is active
model.train()
output_train = model(dummy_input)

# Evaluation mode - dropout is disabled
model.eval()
output_eval = model(dummy_input)

# Outputs will be different because dropout changes behavior!
```

**Critical Rule**: ALWAYS set `model.eval()` before testing/inference!

---

## Common nn.Module Layers

### Fully Connected (Linear)
```python
nn.Linear(in_features, out_features, bias=True)
# Example: nn.Linear(784, 128)
```

### Activation Functions
```python
nn.ReLU()           # Most common: f(x) = max(0, x)
nn.Sigmoid()        # Outputs between 0 and 1
nn.Tanh()           # Outputs between -1 and 1
nn.LeakyReLU()      # ReLU with small negative slope
```

### Regularization
```python
nn.Dropout(p=0.5)        # Randomly zero out p% of neurons
nn.BatchNorm1d(features) # Normalize layer inputs
```

### Loss Functions
```python
nn.CrossEntropyLoss()    # For classification
nn.MSELoss()             # For regression
nn.BCELoss()             # For binary classification
```

---

## Why This Matters for Transformers

The same `nn.Module` structure you learned here is used to build:
- **Attention layers** (the heart of transformers)
- **Multi-head attention blocks**
- **Complete transformer architectures** (GPT, BERT, etc.)

Every transformer is just a sophisticated `nn.Module` with many sub-modules!

---

## Key Takeaways 💡

✅ `nn.Module` is the base class for ALL neural networks in PyTorch
✅ Define layers in `__init__()`, connect them in `forward()`
✅ PyTorch automatically tracks parameters for optimization
✅ Use `model.train()` for training, `model.eval()` for inference
✅ Call `model(x)` to get predictions (don't call `forward()` directly)

**Connection to LLMs**: Every large language model (GPT, LLaMA, etc.) is built using this exact pattern! They're just much larger nn.Modules with specialized attention layers.

---

## Next Steps 🚀

Now you'll learn about **Loss Functions & Optimizers** - the components that actually make your neural network learn from data!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "Which method must you implement when creating a custom nn.Module?",
        "options": [
            "forward()",
            "backward()",
            "train()",
            "predict()"
        ],
        "correct": "forward()",
        "explanation": "The forward() method defines how data flows through your network. PyTorch handles backward() automatically through autograd."
    },
    {
        "question": "What does model.eval() do?",
        "options": [
            "Evaluates the model's performance",
            "Disables training-specific behaviors like dropout",
            "Calculates the loss function",
            "Saves the model to disk"
        ],
        "correct": "Disables training-specific behaviors like dropout",
        "explanation": "model.eval() sets the model to evaluation mode, which disables dropout and changes batch normalization behavior. It doesn't evaluate performance - you still need to run forward passes and calculate metrics."
    },
    {
        "question": "Why don't we apply an activation function on the last layer of a classifier?",
        "options": [
            "It would make training slower",
            "The loss function applies the necessary transformation internally",
            "It would cause gradients to vanish",
            "Activation functions only work on hidden layers"
        ],
        "correct": "The loss function applies the necessary transformation internally",
        "explanation": "For classification, loss functions like CrossEntropyLoss expect raw logits (unnormalized scores) and apply softmax internally for numerical stability. Applying softmax yourself would cause issues."
    },
    {
        "question": "When you call model(x), what actually happens?",
        "options": [
            "It trains the model",
            "It calls model.forward(x) automatically",
            "It calculates the loss",
            "It updates the weights"
        ],
        "correct": "It calls model.forward(x) automatically",
        "explanation": "PyTorch overrides the __call__ method to automatically invoke forward(). This is why we never call forward() directly - we just use model(x)."
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
st.info("👈 Use the sidebar to navigate to the next topic: **04 - Loss Functions & Optimizers**")
