"""
Topic 1: Introduction to PyTorch & Tensors
Level: Basic
"""

from utils.quiz_handler import QuizHandler, create_definition_question, create_code_question

# Topic metadata
TOPIC_ID = "basic_01_pytorch_tensors"
TITLE = "Introduction to PyTorch & Tensors 🔢"
DESCRIPTION = "Learn what PyTorch is, why it was created, and understand the fundamental building block: tensors"

# Main content
CONTENT = """
# Introduction to PyTorch & Tensors 🔢

## What is PyTorch?

PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab (FAIR). It was created to solve several critical problems in deep learning:

### Why PyTorch was Created 🤔

1. **Dynamic Computational Graphs**: Unlike older frameworks (TensorFlow 1.x), PyTorch uses dynamic graphs that are built on-the-fly. This makes debugging easier and allows for more flexible model architectures.

2. **Python-First Design**: PyTorch feels natural to Python developers. If you know Python, you can learn PyTorch quickly!

3. **Research Flexibility**: Researchers needed a framework that could handle experimental architectures easily. PyTorch excels at this.

4. **Automatic Differentiation**: PyTorch automatically computes gradients (derivatives) for you, which is essential for training neural networks.

---

## What are Tensors?

**Tensors are the fundamental data structure in PyTorch.** Think of them as multi-dimensional arrays that can run operations on GPUs for faster computation.

### Why Tensors? 🎯

In deep learning, we work with multi-dimensional data:
- Images: 3D tensors (height × width × color channels)
- Text: 2D tensors (sequence length × embedding dimension)
- Video: 4D tensors (time × height × width × channels)
- Batches: Add an extra dimension for batch size

Tensors allow us to efficiently store and manipulate all this data!

---

## Creating Tensors

Let's see how to create tensors in PyTorch:

```python
import torch

# 1. From a Python list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {tensor_from_list}")

# 2. Tensor with zeros
zeros = torch.zeros(3, 4)  # 3 rows, 4 columns
print(f"\\nZeros tensor:\\n{zeros}")

# 3. Tensor with ones
ones = torch.ones(2, 3)
print(f"\\nOnes tensor:\\n{ones}")

# 4. Random tensor (very common in neural networks!)
random_tensor = torch.randn(2, 3)  # Normal distribution, mean=0, std=1
print(f"\\nRandom tensor:\\n{random_tensor}")

# 5. Tensor with a specific range
range_tensor = torch.arange(0, 10, 2)  # Start, end, step
print(f"\\nRange tensor: {range_tensor}")
```

---

## Tensor Properties

Tensors have important properties you should know:

```python
import torch

# Create a sample tensor
x = torch.randn(3, 4, 5)  # 3D tensor

# Check shape (dimensions)
print(f"Shape: {x.shape}")  # torch.Size([3, 4, 5])

# Check data type
print(f"Data type: {x.dtype}")  # torch.float32

# Check device (CPU or GPU)
print(f"Device: {x.device}")  # cpu

# Number of dimensions
print(f"Dimensions: {x.ndim}")  # 3

# Total number of elements
print(f"Total elements: {x.numel()}")  # 3 × 4 × 5 = 60
```

### Understanding Shape 📏

The shape tells you the size of each dimension:
- `torch.Size([3, 4, 5])` means:
  - Dimension 0: size 3
  - Dimension 1: size 4
  - Dimension 2: size 5

---

## Basic Tensor Operations

```python
import torch

# Create two tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise addition
c = a + b
print(f"Addition: {c}")  # [5., 7., 9.]

# Element-wise multiplication
d = a * b
print(f"Multiplication: {d}")  # [4., 10., 18.]

# Matrix operations
matrix_a = torch.randn(2, 3)
matrix_b = torch.randn(3, 4)

# Matrix multiplication
result = torch.matmul(matrix_a, matrix_b)  # or matrix_a @ matrix_b
print(f"Matrix multiplication shape: {result.shape}")  # [2, 4]
```

---

## CPU vs GPU Tensors 🖥️

PyTorch tensors can live on different devices:

```python
import torch

# Create a tensor on CPU (default)
cpu_tensor = torch.tensor([1, 2, 3])
print(f"Device: {cpu_tensor.device}")  # cpu

# Check if GPU is available
if torch.cuda.is_available():
    # Move tensor to GPU
    gpu_tensor = cpu_tensor.to('cuda')
    print(f"GPU Device: {gpu_tensor.device}")  # cuda:0

    # Move back to CPU
    back_to_cpu = gpu_tensor.to('cpu')
else:
    print("No GPU available, using CPU")
```

**Note**: For this course, we'll use CPU-only examples, but the concepts are the same!

---

## Reshaping Tensors

Changing tensor shapes is common in deep learning:

```python
import torch

# Create a tensor
x = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original: {x.shape}")  # torch.Size([12])

# Reshape to 3x4
x_reshaped = x.reshape(3, 4)
print(f"Reshaped: {x_reshaped.shape}")  # torch.Size([3, 4])

# Reshape to 2x2x3
x_3d = x.reshape(2, 2, 3)
print(f"3D: {x_3d.shape}")  # torch.Size([2, 2, 3])

# Use -1 to infer dimension
x_auto = x.reshape(3, -1)  # PyTorch figures out the second dimension
print(f"Auto: {x_auto.shape}")  # torch.Size([3, 4])
```

---

## Slicing and Indexing

Just like NumPy arrays:

```python
import torch

x = torch.arange(12).reshape(3, 4)
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# First row
print(x[0])  # tensor([0, 1, 2, 3])

# First column
print(x[:, 0])  # tensor([0, 4, 8])

# Specific element
print(x[1, 2])  # tensor(6)

# Slice
print(x[0:2, 1:3])  # First 2 rows, columns 1-2
```

---

## Connection to Deep Learning 🧠

**Why are tensors so important for deep learning?**

1. **Neural Network Inputs**: All data (images, text, audio) is converted to tensors
2. **Parameters**: Weights and biases in neural networks are tensors
3. **Gradients**: During training, gradients are computed as tensors
4. **GPU Acceleration**: Tensors can be moved to GPU for massive speedup
5. **Batch Processing**: We can process multiple examples at once using tensors

**Example**: In an image classifier:
- Input: Tensor of shape `[batch_size, channels, height, width]`
- Weights: Tensors in each layer
- Output: Tensor of class probabilities

Understanding tensors is the foundation for everything else in PyTorch! 🎯

---

## Key Takeaways 💡

1. **PyTorch is designed for flexibility and ease of use** in deep learning research and production
2. **Tensors are multi-dimensional arrays** that hold data in PyTorch
3. **Tensors can live on CPU or GPU** for accelerated computation
4. **Common operations**: creation, reshaping, slicing, and mathematical operations
5. **Everything in deep learning is tensors**: data, parameters, gradients

---

## Next Steps 🚀

In the next topic, we'll learn about **Autograd** - PyTorch's automatic differentiation engine that makes training neural networks possible! This builds directly on tensors.
"""

# Quiz questions
QUESTIONS = [
    QuizHandler.create_multiple_choice(
        question="Why was PyTorch created with dynamic computational graphs?",
        options=[
            "To make debugging easier and allow flexible model architectures",
            "To make models run faster on CPUs",
            "To reduce memory usage",
            "To support only convolutional neural networks"
        ],
        correct_answer="To make debugging easier and allow flexible model architectures",
        explanation="Dynamic graphs are built on-the-fly during execution, making debugging much easier than static graphs. They also allow for more flexible and experimental architectures, which is why researchers love PyTorch!"
    ),

    create_definition_question(
        concept="a tensor in PyTorch",
        correct_definition="A multi-dimensional array that can run operations on GPUs",
        wrong_definitions=[
            "A programming function that trains neural networks",
            "A type of neural network architecture",
            "A GPU acceleration library"
        ],
        explanation="Tensors are the fundamental data structure in PyTorch - they're like NumPy arrays but with GPU support and automatic differentiation capabilities."
    ),

    create_code_question(
        question="Which code correctly creates a 2×3 tensor filled with zeros?",
        code_options=[
            "torch.zeros(2, 3)",
            "torch.zero([2, 3])",
            "torch.empty(2, 3)",
            "torch.tensor([[0, 0, 0], [0, 0, 0]])"
        ],
        correct_code="torch.zeros(2, 3)",
        explanation="torch.zeros(rows, cols) is the standard way to create a tensor filled with zeros. While torch.tensor([[0, 0, 0], [0, 0, 0]]) also works, torch.zeros() is more concise and efficient."
    )
]


def get_topic_content():
    """Return the complete topic content."""
    return {
        'id': TOPIC_ID,
        'title': TITLE,
        'description': DESCRIPTION,
        'content': CONTENT,
        'questions': QUESTIONS
    }
