"""
Topic 2: Autograd & Backpropagation
Level: Basic
"""

from utils.quiz_handler import QuizHandler, create_definition_question, create_why_question

TOPIC_ID = "basic_02_autograd"
TITLE = "Autograd & Backpropagation 🔄"
DESCRIPTION = "Understand automatic differentiation and how PyTorch computes gradients"

CONTENT = """
# Autograd & Backpropagation 🔄

## What is Autograd?

**Autograd** is PyTorch's automatic differentiation engine. It automatically calculates gradients (derivatives) for you - this is the magic that makes training neural networks possible!

### Why Do We Need Autograd? 🤔

In neural networks, we need to:
1. Make predictions (forward pass)
2. Calculate how wrong we were (loss)
3. **Figure out how to adjust weights to improve** (backward pass)

Step 3 requires calculating derivatives of the loss with respect to every parameter. Doing this manually for millions of parameters would be impossible! Autograd does it automatically.

---

## How Autograd Works: Computational Graphs

PyTorch builds a **computational graph** dynamically as you perform operations:

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Perform operations - PyTorch tracks them!
z = x * y + x**2
print(f"z = {z}")  # z = 10.0

# The computational graph is built automatically
# z = x*y + x^2
#   /     \\
#  x*y    x^2
#  / \\    |
# x   y   x
```

### The `requires_grad` Flag 📌

This tells PyTorch: "track operations on this tensor so we can compute gradients later"

```python
import torch

# Tensors that need gradients (usually model parameters)
weights = torch.randn(3, 4, requires_grad=True)
bias = torch.zeros(4, requires_grad=True)

# Tensors that don't need gradients (usually inputs)
data = torch.randn(10, 3)

print(f"Weights requires_grad: {weights.requires_grad}")  # True
print(f"Data requires_grad: {data.requires_grad}")  # False
```

---

## The `.backward()` Method

This is where the magic happens! `.backward()` computes all gradients automatically:

```python
import torch

# Simple example: f(x) = x^2
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2

# Compute gradients
y.backward()

# Gradient is stored in x.grad
# df/dx = 2x = 2*3 = 6
print(f"Gradient: {x.grad}")  # tensor([6.])
```

### Why `.backward()` is Essential

- Manually computing gradients for deep networks = impossibly complex
- `.backward()` uses the **chain rule** automatically
- It traverses the computational graph backwards (hence "backward")

---

## Practical Example: Linear Function

Let's see how gradients flow through a simple function:

```python
import torch

# Linear function: y = 3x + 2
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = 3 * x + 2

# Suppose we have a loss (sum of outputs)
loss = y.sum()

# Compute gradients
loss.backward()

# Gradient of loss with respect to x
# loss = sum(3x + 2) = 3*sum(x) + 6
# d(loss)/dx = 3 for each element
print(f"x.grad: {x.grad}")  # tensor([3., 3., 3.])
```

---

## Chain Rule in Action

Autograd uses the chain rule to backpropagate through multiple operations:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# Multiple operations
a = x * 2        # a = 2x
b = a + 3        # b = 2x + 3
c = b ** 2       # c = (2x + 3)^2
loss = c.sum()

# Backward pass
loss.backward()

# PyTorch computed: d(loss)/dx using chain rule
# dc/dx = dc/db * db/da * da/dx
# = 2(2x+3) * 1 * 2
# = 4(2x+3) = 4(2*2+3) = 28
print(f"Gradient: {x.grad}")  # tensor([28.])
```

### Connection to Deep Learning 🧠

In a neural network:
- Each layer is an operation
- Autograd chains gradients through all layers
- This is how the network learns: gradients tell it how to adjust weights

---

## Gradient Accumulation

**Important**: Gradients accumulate by default!

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"First gradient: {x.grad}")  # tensor([2.])

# Second computation - gradients add up!
y2 = x ** 3
y2.backward()
print(f"Accumulated gradient: {x.grad}")  # tensor([5.]) = 2 + 3

# Always zero gradients before new backward pass
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zeroing: {x.grad}")  # tensor([2.])
```

### Why This Matters

In training loops, you must zero gradients before each backward pass:

```python
# Typical training pattern
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # ← Clear old gradients
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()       # ← Compute new gradients
        optimizer.step()      # ← Update weights
```

---

## No-Grad Context

Sometimes you don't need gradients (e.g., during inference):

```python
import torch

x = torch.tensor([1.0], requires_grad=True)

# Operations tracked
y1 = x * 2
print(f"y1 requires_grad: {y1.requires_grad}")  # True

# Operations NOT tracked
with torch.no_grad():
    y2 = x * 2
    print(f"y2 requires_grad: {y2.requires_grad}")  # False

# Saves memory and speeds up computation!
```

### When to Use `torch.no_grad()`

- **Inference/Evaluation**: Don't need gradients for predictions
- **Metric computation**: Accuracy, precision don't need gradients
- **Preprocessing**: Data transformations

---

## Detaching from the Graph

`.detach()` creates a tensor that shares data but has no gradient tracking:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 2

# Detach y from the computation graph
y_detached = y.detach()

print(f"y requires_grad: {y.requires_grad}")  # True
print(f"y_detached requires_grad: {y_detached.requires_grad}")  # False
```

---

## Higher-Order Gradients

You can even compute gradients of gradients!

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# First derivative
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"dy/dx = {grad1}")  # 3x^2 = 12

# Second derivative
grad2 = torch.autograd.grad(grad1, x)[0]
print(f"d²y/dx² = {grad2}")  # 6x = 12
```

---

## Connection to Neural Networks 🎯

**How autograd enables neural network training:**

1. **Forward Pass**: Input → Layers → Output (graph is built)
2. **Loss Calculation**: Compare output to target
3. **Backward Pass**: `.backward()` computes gradients
4. **Optimization**: Update weights using gradients

```python
import torch
import torch.nn as nn

# Simple neural network
model = nn.Linear(10, 1)  # All parameters have requires_grad=True

# Forward pass (graph is built)
x = torch.randn(5, 10)
output = model(x)

# Backward pass (gradients computed)
loss = output.sum()
loss.backward()

# Gradients are now in model.parameters()
for name, param in model.named_parameters():
    print(f"{name} gradient shape: {param.grad.shape}")
```

---

## Key Takeaways 💡

1. **Autograd = automatic differentiation** - computes gradients for you
2. **Computational graphs** track operations during forward pass
3. **`.backward()`** traverses the graph backward, computing gradients
4. **Chain rule** is applied automatically through all operations
5. **`requires_grad=True`** enables gradient tracking
6. **Always zero gradients** before a new backward pass
7. **`torch.no_grad()`** disables tracking when you don't need gradients

---

## Next Steps 🚀

Now that you understand how PyTorch computes gradients, you're ready to learn about **Neural Networks (nn.Module)** - where we'll use autograd to train actual models!
"""

QUESTIONS = [
    QuizHandler.create_multiple_choice(
        question="Why is autograd essential for training neural networks?",
        options=[
            "It automatically computes gradients for all parameters, making backpropagation possible",
            "It makes the forward pass faster",
            "It reduces memory usage during training",
            "It converts tensors to NumPy arrays"
        ],
        correct_answer="It automatically computes gradients for all parameters, making backpropagation possible",
        explanation="Autograd automatically calculates derivatives using the chain rule, which is essential for backpropagation. Manual gradient computation for millions of parameters would be impractical!"
    ),

    create_definition_question(
        concept="a computational graph in PyTorch",
        correct_definition="A dynamic graph that tracks operations on tensors to enable automatic differentiation",
        wrong_definitions=[
            "A visualization tool for plotting neural network architectures",
            "A static graph defined before execution like in TensorFlow 1.x",
            "A performance profiling tool for PyTorch operations"
        ],
        explanation="PyTorch builds computational graphs dynamically as operations execute. This graph records what operations were performed so gradients can be computed via backpropagation."
    ),

    create_why_question(
        concept="zeroing gradients before each backward pass",
        model_answer="Gradients accumulate by default in PyTorch. If you don't zero them, new gradients will be added to old ones, leading to incorrect gradient values and poor training. Using `optimizer.zero_grad()` or `tensor.grad.zero_()` ensures you start fresh each iteration.",
        explanation="This is one of the most common bugs in PyTorch! Always call `optimizer.zero_grad()` at the start of your training loop."
    )
]

def get_topic_content():
    return {
        'id': TOPIC_ID,
        'title': TITLE,
        'description': DESCRIPTION,
        'content': CONTENT,
        'questions': QUESTIONS
    }
