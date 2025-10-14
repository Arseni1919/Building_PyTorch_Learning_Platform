"""
Topic 7: Custom Datasets & DataLoaders - Intermediate Level
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="07 - Custom Datasets & DataLoaders",
    page_icon="📂",
    layout="wide"
)

# Main content
st.markdown("""
# Custom Datasets & DataLoaders 📂

## Why Custom Datasets?

In real-world projects, you won't always use pre-built datasets like MNIST or CIFAR-10. You'll need to:
- Load your own images from folders
- Process custom data formats (CSV, JSON, etc.)
- Apply specific transformations
- Handle data augmentation

PyTorch provides `Dataset` and `DataLoader` classes to make this easy!

---

## The Dataset Class

A custom dataset must inherit from `torch.utils.data.Dataset` and implement two methods:

### 1. `__len__()` - Return the size of the dataset

```python
def __len__(self):
    return len(self.data)
```

### 2. `__getitem__()` - Get a single sample by index

```python
def __getitem__(self, idx):
    # Load and return the sample at index idx
    return sample, label
```

**Why this structure?** It allows PyTorch to efficiently iterate through your data in batches!

---

## Example 1: Simple Custom Dataset

Let's create a dataset from a list of numbers:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleNumberDataset(Dataset):
    def __init__(self, size=1000):
        # Generate random data
        self.data = torch.randn(size, 10)  # 1000 samples, 10 features
        self.labels = torch.randint(0, 2, (size,))  # Binary labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset
dataset = SimpleNumberDataset(size=1000)

# Check it works
print(f"Dataset size: {len(dataset)}")
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}")
print(f"Label: {label}")
```

Output:
```
Dataset size: 1000
Sample shape: torch.Size([10])
Label: tensor(1)
```

---

## Example 2: Image Dataset from Folder

A common real-world scenario - loading images from folders:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        \"\"\"
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to apply
        \"\"\"
        self.root_dir = root_dir
        self.transform = transform

        # Get all image file paths
        self.image_paths = []
        self.labels = []

        # Assuming structure: root_dir/class_name/image.jpg
        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset(
    root_dir='./my_images',
    transform=transform
)
```

**Directory structure expected:**
```
my_images/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

---

## The DataLoader

`DataLoader` wraps a dataset and provides:
- **Batching**: Combine multiple samples into batches
- **Shuffling**: Randomize order for better training
- **Parallel loading**: Load data in multiple processes
- **Automatic batching**: Handle variable-sized data

```python
from torch.utils.data import DataLoader

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Number of samples per batch
    shuffle=True,       # Shuffle data every epoch
    num_workers=4,      # Use 4 processes for loading (0 for single process)
    pin_memory=True     # Faster GPU transfer (if using GPU)
)

# Iterate through batches
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print(f"  Images shape: {images.shape}")  # [32, 3, 224, 224]
    print(f"  Labels shape: {labels.shape}")  # [32]

    # Your training code here
    # outputs = model(images)
    # loss = criterion(outputs, labels)
    # ...
```

### Key Parameters:

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `batch_size` | Samples per batch | 16, 32, 64, 128 |
| `shuffle` | Randomize order | True (train), False (test) |
| `num_workers` | Parallel processes | 0, 2, 4, 8 |
| `pin_memory` | Faster GPU transfer | True (if GPU available) |
| `drop_last` | Drop incomplete last batch | True (sometimes for training) |

---

## Example 3: CSV Dataset

Loading data from a CSV file:

```python
import torch
from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        \"\"\"
        Args:
            csv_file (str): Path to the CSV file
            transform (callable, optional): Optional transform
        \"\"\"
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get row
        row = self.data_frame.iloc[idx]

        # Extract features and label
        # Assuming last column is label, rest are features
        features = torch.tensor(row[:-1].values, dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label

# Usage
dataset = CSVDataset('data.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Data Transforms and Augmentation

Transforms are crucial for improving model performance:

### Common Image Transforms:

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),           # Random crop for augmentation
    transforms.RandomHorizontalFlip(),    # Flip 50% of images
    transforms.RandomRotation(15),        # Rotate up to 15 degrees
    transforms.ColorJitter(                # Random color changes
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Test transforms (no augmentation!)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # Fixed size, no random
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why augmentation?** It artificially increases dataset size and helps prevent overfitting!

---

## Train/Val/Test Split

Split your dataset properly:

```python
from torch.utils.data import random_split

# Create full dataset
full_dataset = CustomImageDataset(root_dir='./data', transform=transform)

# Split: 70% train, 15% val, 15% test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")
```

---

## Best Practices

### 1. Lazy Loading
Load data only when needed (in `__getitem__`), not in `__init__`:

```python
# ✅ GOOD - Lazy loading
def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx])  # Load when needed
    return image, self.labels[idx]

# ❌ BAD - Eager loading (uses too much memory!)
def __init__(self, image_paths):
    self.images = [Image.open(p) for p in image_paths]  # Loads everything!
```

### 2. Caching for Small Datasets
For small datasets that fit in memory, cache after first load:

```python
class CachedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cache = {}  # Cache dictionary

    def __getitem__(self, idx):
        if idx not in self.cache:
            # Load and cache
            image = Image.open(self.image_paths[idx])
            self.cache[idx] = image

        image = self.cache[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
```

### 3. Error Handling
Handle corrupted files gracefully:

```python
def __getitem__(self, idx):
    try:
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    except Exception as e:
        print(f"Error loading image {idx}: {e}")
        # Return a placeholder or skip
        return self.__getitem__((idx + 1) % len(self))
```

---

## Connection to Transformers

The same DataLoader pattern is used for NLP with transformers:

```python
# Text dataset for transformers
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Usage with transformer models (GPT, BERT, etc.)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## Key Takeaways 💡

✅ **Custom datasets** inherit from `Dataset` and implement `__len__` and `__getitem__`
✅ **DataLoader** handles batching, shuffling, and parallel loading
✅ **Lazy loading** saves memory - load data in `__getitem__`, not `__init__`
✅ **Data augmentation** improves model generalization (training only!)
✅ **Split your data** properly: train, validation, and test sets
✅ **Same pattern** works for images, text, audio, and any custom data

**Next topic**: Learn about Convolutional Neural Networks (CNNs) - the architecture that revolutionized computer vision!
""")

# Quiz section
st.markdown("---")
st.markdown("## 📝 Knowledge Check")

questions = [
    {
        "question": "Which two methods must you implement when creating a custom Dataset?",
        "options": [
            "__len__() and __getitem__()",
            "__init__() and forward()",
            "__len__() and __next__()",
            "__getitem__() and __setitem__()"
        ],
        "correct": "__len__() and __getitem__()",
        "explanation": "A custom Dataset must implement __len__() to return the dataset size and __getitem__() to retrieve a sample by index. These allow PyTorch to iterate through your data efficiently."
    },
    {
        "question": "Why should you use lazy loading (load data in __getitem__) instead of eager loading (load all data in __init__)?",
        "options": [
            "It's faster for small datasets",
            "It saves memory by only loading data when needed",
            "It's required by PyTorch",
            "It improves model accuracy"
        ],
        "correct": "It saves memory by only loading data when needed",
        "explanation": "Lazy loading loads data only when accessed in __getitem__(), which saves memory especially for large datasets. Loading everything in __init__() would consume too much RAM."
    },
    {
        "question": "What is the purpose of data augmentation?",
        "options": [
            "To make training faster",
            "To reduce memory usage",
            "To artificially increase dataset size and prevent overfitting",
            "To normalize the data"
        ],
        "correct": "To artificially increase dataset size and prevent overfitting",
        "explanation": "Data augmentation applies random transformations (flips, rotations, crops) during training to create variations of existing data. This helps the model generalize better and prevents overfitting."
    },
    {
        "question": "Why do we set shuffle=True for training DataLoader but shuffle=False for test DataLoader?",
        "options": [
            "Shuffling makes training faster",
            "Shuffling prevents the model from learning order-dependent patterns; we don't need it for testing",
            "Shuffling is required for backpropagation",
            "Test data is already shuffled"
        ],
        "correct": "Shuffling prevents the model from learning order-dependent patterns; we don't need it for testing",
        "explanation": "Shuffling training data prevents the model from learning spurious patterns based on data order. For testing, we just want consistent evaluation results, so shuffling is unnecessary."
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
st.info("👈 Use the sidebar to navigate to the next topic: **08 - Convolutional Neural Networks**")
