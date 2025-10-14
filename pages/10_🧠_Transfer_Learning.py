"""
Topic 10: Transfer Learning - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="10 - Transfer Learning",
    page_icon="🔄",
    layout="wide"
)

# Main content
st.markdown("""
# Transfer Learning 🔄

## What is Transfer Learning?

**Transfer Learning** means using a model trained on one task as the starting point for a different but related task.

### The Problem: Training from Scratch is Hard

Training a deep CNN from random initialization requires:
- **Millions of labeled images** (like ImageNet: 1.2M images, 1000 classes)
- **Days/weeks of GPU training** (ResNet-50: 14M parameters, 4 GPUs, several days)
- **Lots of hyperparameter tuning**
- **Risk of overfitting** on small datasets

### The Solution: Transfer Learning

Instead of starting from scratch:
1. Take a model pre-trained on a large dataset (ImageNet)
2. Adapt it to your specific task
3. Train on your dataset (much smaller, much faster!)

**Why this works**: Early layers learn general features (edges, textures) that are useful across many tasks!

---

## Feature Hierarchy in CNNs

Pre-trained models learn a hierarchy of features:

```
Layer 1 (Early):          Layer 10 (Middle):       Layer 20 (Late):
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│ Edges       │          │ Textures    │          │ Dogs        │
│ Corners     │   ──>    │ Patterns    │   ──>    │ Cars        │
│ Lines       │          │ Parts       │          │ Faces       │
└─────────────┘          └─────────────┘          └─────────────┘
 Generic features         Mid-level features       Task-specific
 (reusable!)             (somewhat reusable)       (task-specific)
```

**Key insight**: Early layers are generic and transfer well! Late layers are task-specific and need adaptation.

---

## Two Transfer Learning Approaches

### 1. Feature Extraction (Freeze Base Model)

Use pre-trained model as a fixed feature extractor:

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)  # Trained on ImageNet

# Freeze all parameters (no training!)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = model.fc.in_features  # 2048 for ResNet-50
model.fc = nn.Linear(num_features, 10)  # 10 classes for your task

# Only the new FC layer will be trained!
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}")

# Output:
# Trainable parameters:
#   fc.weight
#   fc.bias
```

**When to use**: Small dataset (< 10k images), similar to ImageNet

### 2. Fine-Tuning (Train Entire Model)

Unfreeze some/all layers and train with a small learning rate:

```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# All parameters are trainable (requires_grad=True by default)
# Use different learning rates for different parts!
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},  # Last conv block: small LR
    {'params': model.fc.parameters(), 'lr': 1e-3}        # New FC layer: larger LR
], lr=1e-4)
```

**When to use**: Medium/large dataset (> 10k images), different from ImageNet

---

## Popular Pre-trained Models in PyTorch

PyTorch provides many pre-trained models via `torchvision.models`:

### ResNet Family (Residual Networks)

```python
from torchvision import models

# Different depths available
resnet18 = models.resnet18(pretrained=True)    # 11M params, fastest
resnet34 = models.resnet34(pretrained=True)    # 21M params
resnet50 = models.resnet50(pretrained=True)    # 25M params, best trade-off
resnet101 = models.resnet101(pretrained=True)  # 44M params
resnet152 = models.resnet152(pretrained=True)  # 60M params, most accurate

# All trained on ImageNet (1000 classes)
```

**Best for**: General-purpose computer vision, great accuracy/speed trade-off

### VGG Family (Very Deep Networks)

```python
vgg16 = models.vgg16(pretrained=True)   # 138M params
vgg19 = models.vgg19(pretrained=True)   # 144M params
```

**Best for**: Style transfer, feature extraction (but very large!)

### EfficientNet Family (Efficient Scaling)

```python
efficientnet_b0 = models.efficientnet_b0(pretrained=True)  # 5M params, efficient
efficientnet_b7 = models.efficientnet_b7(pretrained=True)  # 66M params, accurate
```

**Best for**: Mobile/embedded devices, limited compute

### MobileNet Family (Mobile-Optimized)

```python
mobilenet_v2 = models.mobilenet_v2(pretrained=True)  # 3.5M params
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)  # 2.5M params
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)  # 5.5M params
```

**Best for**: Real-time mobile applications, very fast inference

---

## Complete Transfer Learning Example

### Step 1: Load Pre-trained Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
print(f"Original model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Check the final layer
print(f"Original final layer: {model.fc}")
# Linear(in_features=2048, out_features=1000, bias=True)  # 1000 ImageNet classes
```

### Step 2: Modify for Your Task

```python
# Replace final layer for your number of classes
num_classes = 5  # Your custom dataset has 5 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

print(f"New final layer: {model.fc}")
# Linear(in_features=2048, out_features=5, bias=True)  # 5 classes
```

### Step 3: Freeze Layers (Feature Extraction)

```python
# Freeze all layers except the final FC layer
for name, param in model.named_parameters():
    if 'fc' not in name:  # Don't freeze the final layer
        param.requires_grad = False

# Check what's trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
# Only ~10k parameters (just the final layer!)
```

### Step 4: Prepare Data

```python
# ImageNet normalization (IMPORTANT: use same as pre-training!)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

# Load your custom dataset
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder('path/to/train', transform=transform)
val_dataset = ImageFolder('path/to/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Step 5: Train

```python
# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only train FC layer

# Training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
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

# Train for a few epochs
num_epochs = 5
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
```

---

## Fine-Tuning: Unfreezing Layers

After initial training with frozen layers, fine-tune the entire model:

### Progressive Unfreezing

```python
# Step 1: Train only the final layer (done above)

# Step 2: Unfreeze last conv block
for param in model.layer4.parameters():  # Last residual block
    param.requires_grad = True

optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},  # Small LR for pre-trained
    {'params': model.fc.parameters(), 'lr': 1e-4}        # Larger LR for new layer
])

# Train for a few more epochs
# ...

# Step 3: Unfreeze entire model (optional)
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Very small LR

# Fine-tune for a few more epochs
# ...
```

**Why progressive?** Prevents catastrophic forgetting of pre-trained features!

---

## Different Learning Rates for Different Layers

Use smaller learning rates for pre-trained layers:

```python
import torch.optim as optim

# Option 1: Different LR per layer group
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-6},   # Early layers: tiny LR
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-5},
    {'params': model.layer4.parameters(), 'lr': 1e-4},   # Late layers: small LR
    {'params': model.fc.parameters(), 'lr': 1e-3}        # New layer: normal LR
])

# Option 2: Discriminative learning rates
base_lr = 1e-5
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': base_lr / 10},
    {'params': model.layer2.parameters(), 'lr': base_lr / 5},
    {'params': model.layer3.parameters(), 'lr': base_lr / 2},
    {'params': model.layer4.parameters(), 'lr': base_lr},
    {'params': model.fc.parameters(), 'lr': base_lr * 10}
])
```

**Rule of thumb**: Early layers (generic features) need tiny LR, late layers (task-specific) need larger LR!

---

## When to Use Transfer Learning vs Training from Scratch

### Use Transfer Learning When:

✅ Your dataset is small (< 100k images)
✅ Your task is similar to ImageNet (natural images, object recognition)
✅ You have limited compute resources
✅ You need results quickly
✅ You want better generalization

### Train from Scratch When:

✅ Your dataset is very large (> 1M images)
✅ Your task is very different from ImageNet (medical images, satellite images, drawings)
✅ You have lots of compute and time
✅ Domain-specific architectures exist (medical imaging, etc.)

### Decision Tree:

```
Is your dataset similar to ImageNet (natural images)?
│
├─ YES: Is your dataset small (< 10k)?
│   ├─ YES: Feature extraction (freeze all, train FC only)
│   └─ NO: Fine-tuning (train entire model with small LR)
│
└─ NO: Is your dataset large (> 100k)?
    ├─ YES: Train from scratch OR fine-tune with aggressive LR
    └─ NO: Try transfer learning anyway (generic features help!)
```

---

## Practical Tips for Transfer Learning

### 1. Use the Same Preprocessing

**CRITICAL**: Use the same normalization as pre-training!

```python
# ImageNet models expect this normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Apply to your images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize  # ✅ Must match pre-training!
])
```

### 2. Start with Feature Extraction

```python
# Step 1: Freeze and train FC only (fast, few epochs)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Train for 5-10 epochs

# Step 2: Fine-tune last block (slow, more epochs)
for param in model.layer4.parameters():
    param.requires_grad = True
# Train for 10-20 epochs with small LR

# Step 3: Fine-tune entire model (optional)
for param in model.parameters():
    param.requires_grad = True
# Train for 5-10 epochs with very small LR
```

### 3. Use Data Augmentation

Even with transfer learning, augmentation helps:

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 4. Monitor for Overfitting

```python
# Track train vs validation accuracy
train_acc = 95%
val_acc = 70%  # ❌ Overfitting!

# Solutions:
# - Increase dropout
# - Add more data augmentation
# - Reduce learning rate
# - Freeze more layers
```

---

## Connection to LLM Fine-Tuning

Transfer learning for images is conceptually identical to LLM fine-tuning:

### Computer Vision (CNNs):
```python
# Pre-train on ImageNet (1.2M images, 1000 classes)
model = models.resnet50(pretrained=True)

# Fine-tune on custom dataset (10k images, 5 classes)
model.fc = nn.Linear(2048, 5)
# Train with small LR
```

### Natural Language Processing (Transformers):
```python
# Pre-train on massive text corpus (billions of tokens)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Fine-tune on custom task (sentiment analysis, few thousand examples)
model.classifier = nn.Linear(768, 2)  # Binary sentiment
# Train with small LR
```

**Same idea**: Pre-training learns general representations, fine-tuning adapts to specific tasks!

### Modern LLM Techniques:

- **Full fine-tuning**: Update all parameters (like fine-tuning CNNs)
- **LoRA (Low-Rank Adaptation)**: Only train small adapter modules (like feature extraction)
- **Prompt tuning**: Only optimize input embeddings (even cheaper!)

Transfer learning in vision paved the way for modern LLM training strategies!

---

## Complete Working Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Prepare data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # Print metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")

    return model

print("Transfer learning model ready for training!")
```

---

## Key Takeaways 💡

✅ **Transfer learning**: Use pre-trained models to save time and improve performance
✅ **Two approaches**: Feature extraction (freeze base) or fine-tuning (train all)
✅ **Why it works**: Early layers learn generic features useful across tasks
✅ **Popular models**: ResNet, EfficientNet, MobileNet from torchvision.models
✅ **Key technique**: Freeze layers with `param.requires_grad = False`
✅ **Learning rates**: Small LR for pre-trained layers, larger LR for new layers
✅ **Progressive unfreezing**: Train FC → unfreeze last block → unfreeze all
✅ **Connection to LLMs**: Same concept as fine-tuning BERT, GPT, etc.!

**Next topic**: Learn about Advanced Optimizers and Schedulers - techniques to train models faster and achieve better convergence!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What is the main advantage of transfer learning over training from scratch?",
        "options": [
            "It always gives better accuracy",
            "It requires much less data and training time by leveraging pre-trained features",
            "It uses less memory",
            "It doesn't require a GPU"
        ],
        "correct": "It requires much less data and training time by leveraging pre-trained features",
        "explanation": "Transfer learning leverages features learned on large datasets (like ImageNet's 1.2M images). Early layers learn generic features (edges, textures) that transfer well to new tasks. This means you can achieve good results with much less data (10k instead of 1M images) and training time (hours instead of days)."
    },
    {
        "question": "What does param.requires_grad = False do in transfer learning?",
        "options": [
            "Deletes the parameter from the model",
            "Freezes the parameter so it won't be updated during training",
            "Reduces the parameter to zero",
            "Converts the parameter to CPU"
        ],
        "correct": "Freezes the parameter so it won't be updated during training",
        "explanation": "Setting requires_grad=False tells PyTorch not to compute gradients for that parameter. This 'freezes' the parameter - its values won't change during training. This is used in feature extraction to keep pre-trained weights fixed while only training new layers."
    },
    {
        "question": "When fine-tuning a pre-trained model, why should pre-trained layers use smaller learning rates than new layers?",
        "options": [
            "Pre-trained layers have fewer parameters",
            "Pre-trained layers already have good features; large LR could destroy them",
            "Pre-trained layers train faster",
            "It's just a convention with no real benefit"
        ],
        "correct": "Pre-trained layers already have good features; large LR could destroy them",
        "explanation": "Pre-trained layers already learned useful features on large datasets. Using a large learning rate could catastrophically forget these features. New layers (randomly initialized) need larger LR to learn quickly. This is called discriminative learning rates - early layers get tiny LR (1e-6), late layers get small LR (1e-4), new layers get normal LR (1e-3)."
    },
    {
        "question": "What's the recommended first step when using transfer learning?",
        "options": [
            "Immediately fine-tune the entire model",
            "Freeze all layers except the final FC layer and train just that first",
            "Train from scratch for a few epochs first",
            "Remove all pre-trained weights"
        ],
        "correct": "Freeze all layers except the final FC layer and train just that first",
        "explanation": "Best practice: (1) Freeze base model, train only new FC layer (fast, few epochs). (2) Unfreeze last block, fine-tune with small LR. (3) Optionally unfreeze entire model with very small LR. This progressive approach prevents catastrophic forgetting and is faster than fine-tuning everything from the start."
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
st.info("👈 Use the sidebar to navigate to the next topic: **11 - Advanced Optimizers and Schedulers**")
