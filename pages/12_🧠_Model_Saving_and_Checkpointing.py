"""
Topic 12: Model Saving & Checkpointing - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="12 - Model Saving & Checkpointing",
    page_icon="💾",
    layout="wide"
)

# Main content
st.markdown("""
# Model Saving & Checkpointing 💾

## Why Save Models?

Training deep learning models takes time and resources:

```
Training a large model:
- Hours to days of GPU time
- Thousands of dollars in compute costs
- Careful hyperparameter tuning
```

You need to save models for:
1. **Resume training** after crashes or interruptions
2. **Deploy** to production (inference servers)
3. **Share** with collaborators or the community
4. **Experiment** with different configurations
5. **Reproduce** results from papers

---

## Two Ways to Save Models

### Method 1: Save state_dict() (Recommended ✅)

Save only the model parameters (weights and biases):

```python
import torch

# Save
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MyModel()  # Create model architecture first
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode
```

**Pros**:
- Smaller file size (only parameters)
- More flexible (can load into modified architecture)
- More portable (architecture defined in code)
- **Recommended approach**

**Cons**:
- Must recreate model architecture separately

### Method 2: Save Entire Model (Not Recommended ❌)

Save the entire model object:

```python
# Save
torch.save(model, 'model_full.pth')

# Load
model = torch.load('model_full.pth')
model.eval()
```

**Pros**:
- Simple, one line to save and load

**Cons**:
- Larger file size
- Less flexible (breaks if architecture changes)
- Can break with PyTorch version changes
- Serializes Python classes (security risk)
- **Not recommended for production**

---

## Saving and Loading: Complete Example

### Saving a Model

```python
import torch
import torch.nn as nn

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train model
model = SimpleModel()
# ... training code ...

# ✅ Save the state_dict (recommended)
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")
```

### Loading a Model

```python
import torch
import torch.nn as nn

# Define the SAME model architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleModel()

# Load saved weights
model.load_state_dict(torch.load('model.pth'))

# Set to evaluation mode (disable dropout, batchnorm training mode)
model.eval()

# Now you can use the model for inference!
with torch.no_grad():
    output = model(input_tensor)
```

---

## Checkpointing: Saving Training Progress

**Checkpointing** saves not just the model, but the entire training state:

### What to Save in a Checkpoint:

```python
checkpoint = {
    'epoch': epoch,                          # Current epoch
    'model_state_dict': model.state_dict(),  # Model weights
    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
    'scheduler_state_dict': scheduler.state_dict(),  # Scheduler state (optional)
    'loss': loss,                            # Current loss
    'best_val_loss': best_val_loss,          # Best validation loss so far
}

torch.save(checkpoint, 'checkpoint.pth')
```

**Why save optimizer state?** Adam/AdamW maintain momentum buffers - resuming without them means starting optimizer from scratch!

### Complete Checkpointing Example:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Training setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_val_loss = float('inf')

# Load checkpoint if exists
checkpoint_path = 'checkpoint.pth'
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    print(f"Resumed from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for images, labels in train_loader:
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save checkpoint every epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, 'checkpoint.pth')

    # Save best model separately
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✅ Saved new best model!")
```

---

## Handling Device Mismatches (CPU ↔ GPU)

### Saving on GPU, Loading on CPU:

```python
# Save on GPU
model = model.to('cuda')
torch.save(model.state_dict(), 'model.pth')

# Load on CPU
model = MyModel()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()
```

### Saving on CPU, Loading on GPU:

```python
# Save on CPU
torch.save(model.state_dict(), 'model.pth')

# Load on GPU
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model = model.to('cuda')
model.eval()
```

### Generic Solution (Handles Both):

```python
# Load and automatically map to current device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel()
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()
```

**Always use `map_location`** when loading to avoid device errors!

---

## Best Practices for Checkpointing

### 1. Save Multiple Checkpoints

Don't overwrite the same checkpoint every time:

```python
# Save with epoch number
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Keep last N checkpoints
max_checkpoints = 5
checkpoints = sorted(glob.glob('checkpoint_epoch_*.pth'))
if len(checkpoints) > max_checkpoints:
    os.remove(checkpoints[0])  # Remove oldest
```

### 2. Save Best Model Separately

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss
        }, 'best_model.pth')
```

### 3. Include Metadata

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),

    # Metadata
    'model_config': {
        'num_layers': 12,
        'hidden_dim': 768,
        'num_classes': 10
    },
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_decay': 0.01
    },
    'timestamp': datetime.now().isoformat(),
    'pytorch_version': torch.__version__
}

torch.save(checkpoint, 'checkpoint_with_metadata.pth')
```

### 4. Checkpoint Periodically

```python
# Save every N epochs
save_interval = 5

for epoch in range(num_epochs):
    # ... training ...

    if (epoch + 1) % save_interval == 0:
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
        print(f"Checkpoint saved at epoch {epoch}")
```

### 5. Verify Checkpoints After Saving

```python
# Save checkpoint
torch.save(checkpoint, 'checkpoint.pth')

# Verify it loads correctly
try:
    loaded = torch.load('checkpoint.pth')
    print("✅ Checkpoint verified!")
except Exception as e:
    print(f"❌ Checkpoint corrupted: {e}")
```

---

## Model Versioning for Production

### Semantic Versioning

```python
model_version = "v1.2.3"
# v1 = major (breaking changes)
# .2 = minor (new features)
# .3 = patch (bug fixes)

checkpoint = {
    'version': model_version,
    'model_state_dict': model.state_dict(),
    'training_config': config
}

torch.save(checkpoint, f'model_{model_version}.pth')
```

### Git Integration

```python
import subprocess

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        return commit
    except:
        return None

checkpoint = {
    'model_state_dict': model.state_dict(),
    'git_commit': get_git_commit(),
    'date': datetime.now().isoformat()
}

torch.save(checkpoint, 'model.pth')
```

---

## Saving Models for Different Purposes

### For Training Resumption:

```python
# Save everything needed to resume
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'rng_state': torch.get_rng_state(),  # For reproducibility
    'train_loss': train_loss,
    'val_loss': val_loss
}

torch.save(checkpoint, 'training_checkpoint.pth')
```

### For Inference/Deployment:

```python
# Save only what's needed for inference
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'num_classes': 10,
        'input_size': 224
    }
}, 'model_for_deployment.pth')

# Even simpler: just state_dict
torch.save(model.state_dict(), 'model_weights.pth')
```

### For Sharing/Open Source:

```python
# Include comprehensive metadata
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'ResNet-50',
    'num_classes': 1000,
    'training_dataset': 'ImageNet-1k',
    'accuracy': 76.5,
    'pytorch_version': torch.__version__,
    'paper': 'https://arxiv.org/abs/1512.03385',
    'license': 'MIT'
}, 'model_for_sharing.pth')
```

---

## Loading Pre-trained Models from torchvision

PyTorch makes loading popular models easy:

```python
from torchvision import models

# Load with pre-trained weights
resnet = models.resnet50(pretrained=True)

# Or with newer API (PyTorch 1.13+)
from torchvision.models import ResNet50_Weights
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Load without weights (random initialization)
resnet = models.resnet50(pretrained=False)
```

### Behind the Scenes:

```python
# torchvision downloads and caches model weights
# Default location: ~/.cache/torch/hub/checkpoints/

# You can change cache directory
torch.hub.set_dir('/path/to/cache')
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting model.eval()

```python
# ❌ BAD - dropout and batchnorm in training mode
model.load_state_dict(torch.load('model.pth'))
output = model(input)  # Wrong behavior!

# ✅ GOOD - set to evaluation mode
model.load_state_dict(torch.load('model.pth'))
model.eval()
output = model(input)  # Correct behavior!
```

### Pitfall 2: Device Mismatch

```python
# ❌ BAD - fails if saved on GPU, loading on CPU
model.load_state_dict(torch.load('model.pth'))

# ✅ GOOD - handle device automatically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
```

### Pitfall 3: Architecture Mismatch

```python
# ❌ BAD - architecture changed
class OldModel(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(784, 10)

# Saved as OldModel, but trying to load as NewModel
class NewModel(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(784, 20)  # Different size!

model = NewModel()
model.load_state_dict(torch.load('model.pth'))  # ❌ Error!

# ✅ GOOD - use same architecture or handle gracefully
try:
    model.load_state_dict(torch.load('model.pth'), strict=False)
    print("Loaded with warnings")
except:
    print("Architecture incompatible")
```

### Pitfall 4: Not Saving Optimizer State

```python
# ❌ BAD - optimizer state lost
torch.save(model.state_dict(), 'model.pth')
# Resume training: optimizer starts from scratch!

# ✅ GOOD - save optimizer too
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpoint.pth')
```

---

## Connection to Modern LLM Practices

Large language models use similar checkpointing strategies:

### Hugging Face Transformers:

```python
from transformers import AutoModel, AutoTokenizer

# Save model and tokenizer
model.save_pretrained('./my_model')
tokenizer.save_pretrained('./my_model')

# Load model and tokenizer
model = AutoModel.from_pretrained('./my_model')
tokenizer = AutoTokenizer.from_pretrained('./my_model')
```

### Checkpoint Sharding (Large Models):

```python
# For models too large for single file
# Shards: pytorch_model-00001-of-00010.bin, etc.

from transformers import AutoModel

# Automatically handles sharded checkpoints
model = AutoModel.from_pretrained('gpt-large-model')
```

### Gradient Checkpointing (Memory Efficiency):

```python
# Different from model checkpointing!
# Trades compute for memory during training

from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Recompute activations instead of storing
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

---

## Complete Production Example

```python
import torch
import os
from datetime import datetime

class ModelCheckpointer:
    def __init__(self, model, optimizer, save_dir='checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.best_val_loss = float('inf')

        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save epoch checkpoint
        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model with val_loss={val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1

# Usage
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
checkpointer = ModelCheckpointer(model, optimizer)

# Resume from checkpoint
start_epoch = checkpointer.load_checkpoint('checkpoints/checkpoint_latest.pth')

# Training loop
for epoch in range(start_epoch, num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    is_best = val_loss < checkpointer.best_val_loss
    if is_best:
        checkpointer.best_val_loss = val_loss

    checkpointer.save_checkpoint(epoch, train_loss, val_loss, is_best)
```

---

## Key Takeaways 💡

✅ **Use state_dict()**: Save model.state_dict(), not entire model
✅ **Checkpointing**: Save model + optimizer + scheduler for resuming
✅ **Best model**: Save best performing model separately from checkpoints
✅ **Device handling**: Use map_location when loading to handle CPU/GPU
✅ **model.eval()**: Always call before inference to disable dropout/batchnorm
✅ **Metadata**: Include version, config, hyperparameters in checkpoints
✅ **Multiple checkpoints**: Keep last N checkpoints, don't overwrite
✅ **Production**: Version models, verify after saving, include git commit

**Next topic**: Learn about Introduction to Embeddings - how neural networks represent discrete data like words!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "What is the recommended way to save a PyTorch model?",
        "options": [
            "torch.save(model, 'model.pth')",
            "torch.save(model.state_dict(), 'model.pth')",
            "pickle.dump(model, file)",
            "model.save('model.pth')"
        ],
        "correct": "torch.save(model.state_dict(), 'model.pth')",
        "explanation": "Saving model.state_dict() is recommended because it only saves parameters (smaller file), is more portable across PyTorch versions, and allows loading into modified architectures. Saving the entire model object is less flexible and can break with version changes."
    },
    {
        "question": "Why should you save the optimizer state in a checkpoint?",
        "options": [
            "To save disk space",
            "To make loading faster",
            "To preserve momentum buffers and other optimizer state for smooth training resumption",
            "It's not necessary to save optimizer state"
        ],
        "correct": "To preserve momentum buffers and other optimizer state for smooth training resumption",
        "explanation": "Optimizers like Adam and SGD with momentum maintain internal state (momentum buffers, adaptive learning rates per parameter). Without saving this, resuming training means starting the optimizer from scratch, which can cause training instability and worse convergence. Always save optimizer_state_dict() in checkpoints!"
    },
    {
        "question": "What does map_location='cpu' do when loading a model?",
        "options": [
            "Moves the model to CPU after loading",
            "Tells PyTorch to map saved tensors to CPU memory during loading",
            "Optimizes the model for CPU",
            "Removes GPU-specific operations"
        ],
        "correct": "Tells PyTorch to map saved tensors to CPU memory during loading",
        "explanation": "map_location specifies where to load tensors. If a model was saved on GPU but you're loading on CPU, without map_location PyTorch will try to load to GPU (which doesn't exist) and fail. map_location='cpu' or map_location=device handles this gracefully by loading to the specified device."
    },
    {
        "question": "What must you do before using a loaded model for inference?",
        "options": [
            "Call model.train()",
            "Call model.eval()",
            "Reset the optimizer",
            "Nothing, it's ready to use"
        ],
        "correct": "Call model.eval()",
        "explanation": "You must call model.eval() to set the model to evaluation mode. This disables dropout (which randomly zeros neurons) and sets batch normalization to use running statistics instead of batch statistics. Without eval(), dropout will give inconsistent predictions and batchnorm will behave incorrectly. Always model.eval() for inference!"
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
st.info("👈 Use the sidebar to navigate to the next topic: **13 - Introduction to Embeddings**")
