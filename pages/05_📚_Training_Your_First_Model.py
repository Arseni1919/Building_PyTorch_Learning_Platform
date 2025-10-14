"""
Topic 5: Training Your First Model - Basic Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="05 - Training Your First Model",
    page_icon="🚀",
    layout="wide"
)

# Main content
st.markdown("""
# Training Your First Model 🚀

## The Complete Training Pipeline

Now we'll combine everything you've learned to train a real neural network! We'll build a classifier for Fashion-MNIST (clothing images).

---

## Dataset: Fashion-MNIST

Fashion-MNIST has:
- **60,000 training images** of clothing items
- **10,000 test images**
- **10 classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **28x28 grayscale images** (just like regular MNIST)

Why Fashion-MNIST? It's more realistic than digit recognition but still simple enough to run on CPU!

---

## Step 1: Load the Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: Convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Download and load training data
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders (iterate through data in batches)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

**Why normalize?** Neural networks train better when inputs are centered around zero.

---

## Step 2: Define the Model

```python
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Flatten 28x28 to 784
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)        # 10 output classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten image
        x = x.view(x.size(0), -1)  # [batch, 28, 28] -> [batch, 784]

        # Forward pass
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on last layer!
        return x

# Create model instance
model = FashionClassifier()
print(model)
```

---

## Step 3: Setup Loss and Optimizer

```python
# Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer (Adam with default learning rate)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if GPU is available (optional - works on CPU too!)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Training on: {device}")
```

---

## Step 4: Training Loop

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    \"\"\"Train for one epoch\"\"\"
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass + optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100 * correct / total:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc
```

**Key points**:
- `model.train()` enables dropout
- We track both loss and accuracy
- `torch.max(outputs, 1)` gets predicted class indices

---

## Step 5: Evaluation Function

```python
def evaluate(model, loader, criterion, device):
    \"\"\"Evaluate model on validation/test set\"\"\"
    model.eval()  # Set to evaluation mode (disables dropout)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass only
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
```

**Critical differences from training**:
- `model.eval()` disables dropout/batch norm training behavior
- `torch.no_grad()` saves memory by not storing gradients

---

## Step 6: Full Training Loop

```python
# Training parameters
num_epochs = 10

print("Starting training...")
for epoch in range(num_epochs):
    print(f"\\nEpoch {epoch + 1}/{num_epochs}")

    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    # Evaluate on test set
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

print("\\nTraining complete!")
```

Expected output (after 10 epochs):
```
Epoch 10/10
  Batch 100/938, Loss: 0.3421, Acc: 87.23%
  ...
Train Loss: 0.3156, Train Acc: 88.51%
Test Loss: 0.3582, Test Acc: 87.09%
```

**You just trained a neural network!** 🎉

---

## Step 7: Make Predictions

```python
# Get one batch of test images
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Make predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)

# Class names
classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show first 5 predictions
for i in range(5):
    print(f"Image {i + 1}: Predicted={classes[predictions[i]]}, "
          f"Actual={classes[labels[i]]}")
```

Output:
```
Image 1: Predicted=Ankle boot, Actual=Ankle boot ✓
Image 2: Predicted=Pullover, Actual=Pullover ✓
Image 3: Predicted=Trouser, Actual=Trouser ✓
Image 4: Predicted=Trouser, Actual=Trouser ✓
Image 5: Predicted=Shirt, Actual=Coat ✗
```

---

## Common Training Issues and Solutions

### 1. Loss Not Decreasing

**Problem**: Loss stays high or increases

**Solutions**:
- ✅ Check learning rate (try 0.001 for Adam)
- ✅ Verify data normalization
- ✅ Check for bugs in forward() method
- ✅ Make sure you're calling `optimizer.zero_grad()`

### 2. Overfitting (Train Acc >> Test Acc)

**Problem**: Model memorizes training data but doesn't generalize

**Solutions**:
- ✅ Add dropout (`nn.Dropout(0.2)`)
- ✅ Use data augmentation
- ✅ Train for fewer epochs
- ✅ Add weight decay (`optimizer = Adam(..., weight_decay=0.01)`)

### 3. Slow Training

**Solutions**:
- ✅ Increase batch size
- ✅ Use GPU if available
- ✅ Simplify model (fewer layers/neurons)

---

## Training Loop Template (Copy-Paste Ready!)

```python
# Complete training loop template
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Calculate metrics...
```

---

## Connection to Transformers

The training loop you just learned is THE SAME for transformers! The only differences:
- Different model architecture (attention layers instead of fully connected)
- Different data (text tokens instead of images)
- Larger scale (billions of parameters, trillions of tokens)

Example LLM training:
```python
# Same structure!
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        input_ids, targets = batch
        outputs = model(input_ids)  # Transformer forward pass
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Key Takeaways 💡

✅ Training loop has 4 steps: forward, loss, backward, optimize
✅ Always call `model.train()` before training, `model.eval()` before testing
✅ Use `torch.no_grad()` during evaluation to save memory
✅ Track both training and test metrics to detect overfitting
✅ The training loop structure is universal (works for CNNs, RNNs, Transformers!)

**Next step**: Learn how to properly evaluate model performance with metrics!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What should you do BEFORE evaluating your model on test data?",
        "options": [
            "Call model.train()",
            "Call model.eval()",
            "Call optimizer.zero_grad()",
            "Call loss.backward()"
        ],
        "correct": "Call model.eval()",
        "explanation": "You must call model.eval() to disable dropout and put batch normalization in inference mode. This ensures your model behaves correctly during evaluation."
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
        "question": "If your training loss is decreasing but test loss is increasing, what problem are you facing?",
        "options": [
            "Underfitting",
            "Overfitting",
            "Wrong learning rate",
            "Vanishing gradients"
        ],
        "correct": "Overfitting",
        "explanation": "When training loss decreases but test loss increases, the model is overfitting - it's memorizing the training data instead of learning generalizable patterns. Solutions include dropout, regularization, or early stopping."
    },
    {
        "question": "Why do we use torch.no_grad() during evaluation?",
        "options": [
            "To speed up forward pass",
            "To disable gradient computation and save memory",
            "To prevent the model from updating weights",
            "To enable dropout"
        ],
        "correct": "To disable gradient computation and save memory",
        "explanation": "torch.no_grad() disables gradient computation, which saves memory and speeds up computation since we don't need gradients when we're only making predictions, not training."
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
st.info("👈 Use the sidebar to navigate to the next topic: **06 - Evaluation & Metrics**")
