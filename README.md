# PyTorch Mastery Course: From Beginner to Advanced (by Claude Code)

A comprehensive, hands-on PyTorch course that takes you from fundamentals to state-of-the-art techniques through practical coding exercises. Learn by building, understanding the "why" behind every concept, and connecting the dots across the deep learning ecosystem.

## Course Philosophy

This course follows a **top-down learning approach**:
- Start with the big picture before diving into implementation details
- Understand **why** each concept exists before learning **how** to implement it
- See **interconnections** between topics and how they fit into the broader ecosystem
- Learn through **hands-on coding** with real-world examples
- Progress from simple to complex with **step-by-step guidance**

## What You'll Learn

By completing this course, you will be able to:
- Implement any PyTorch concept from scratch
- Understand when and why to use different techniques
- Build modern deep learning architectures (including Transformers)
- Debug and optimize PyTorch models effectively
- Apply latest PyTorch features (torch.compile, FlexAttention, etc.)

## Course Structure

The course is divided into **3 levels** with **15 topics total**, progressing from fundamentals to cutting-edge techniques.

### Level 1: Beginner (5 Topics)

Build a solid foundation in PyTorch fundamentals and basic neural networks.

1. **[PyTorch Fundamentals: Tensors & Operations](notebooks/01_beginner/01_pytorch_fundamentals.ipynb)**
   - What are tensors and why do we need them?
   - Tensor operations, shapes, and broadcasting
   - GPU acceleration basics
   - PyTorch vs NumPy: when to use each

2. **[Automatic Differentiation & Backpropagation](notebooks/01_beginner/02_autograd_backprop.ipynb)**
   - Why automatic differentiation is revolutionary
   - How PyTorch tracks gradients
   - Understanding the computational graph
   - Implementing backpropagation from scratch

3. **[Building Neural Networks with nn.Module](notebooks/01_beginner/03_neural_networks.ipynb)**
   - Architecture of neural networks
   - Why nn.Module is the foundation of PyTorch
   - Building custom layers and models
   - Forward pass and parameter management

4. **[Loss Functions: Understanding What to Optimize](notebooks/01_beginner/04_loss_functions.ipynb)**
   - What loss functions measure and why they matter
   - Classification losses (CrossEntropy, BCE, Focal Loss)
   - Regression losses (MSE, MAE, Huber)
   - When to use each loss function (comprehensive guide)

5. **[Training Loop & Optimization](notebooks/01_beginner/05_training_optimization.ipynb)**
   - Anatomy of a training loop
   - Optimizers: SGD, Adam, AdamW - when to use each
   - Learning rate scheduling
   - Validation and model evaluation

### Level 2: Intermediate (5 Topics)

Master core deep learning implementations and modern architectures.

6. **[Convolutional Neural Networks (CNNs)](notebooks/02_intermediate/06_cnns.ipynb)**
   - Why convolutions work for spatial data
   - Building blocks: Conv2d, Pooling, Batch Normalization
   - Classic architectures: ResNet, VGG concepts
   - Implementing a CNN classifier from scratch

7. **[Attention Mechanisms: The Foundation](notebooks/02_intermediate/07_attention_mechanisms.ipynb)**
   - Why attention changed everything
   - Self-attention: querying your own input
   - Cross-attention: relating two sequences
   - Multi-head attention: parallel attention pathways
   - Implementing attention from scratch

8. **[Positional Encoding & Embeddings](notebooks/02_intermediate/08_positional_encoding.ipynb)**
   - Why position matters in sequences
   - Sinusoidal positional encoding (original Transformer)
   - Learned positional embeddings
   - Rotary Position Embeddings (RoPE) - modern approach
   - When to use each type

9. **[The Transformer Architecture](notebooks/02_intermediate/09_transformer_architecture.ipynb)**
   - Big picture: encoder-decoder architecture
   - Building encoder blocks from scratch
   - Building decoder blocks from scratch
   - Putting it all together: full Transformer
   - Why Transformers dominate modern AI

10. **[Advanced Data Loading & Augmentation](notebooks/02_intermediate/10_data_loading.ipynb)**
    - Custom Dataset and DataLoader design
    - Efficient data pipelines
    - Data augmentation techniques
    - Handling imbalanced datasets

### Level 3: Advanced/Professional (5 Topics)

Master state-of-the-art techniques and production-ready implementations.

11. **[Flash Attention: Optimizing Attention](notebooks/03_advanced/11_flash_attention.ipynb)**
    - Why standard attention is memory-inefficient
    - Flash Attention algorithm explained
    - Using PyTorch's FlexAttention API (2.5+)
    - cuDNN Fused Flash Attention on H100
    - Performance benchmarks and when to use it

12. **[Grouped Query Attention (GQA)](notebooks/03_advanced/12_grouped_query_attention.ipynb)**
    - Evolution: Multi-Head � Multi-Query � Grouped Query
    - Why GQA is the sweet spot for efficiency
    - Implementing GQA from scratch
    - Using GQA in modern LLMs (Llama, Mistral)

13. **[Mixture of Experts (MoE)](notebooks/03_advanced/13_mixture_of_experts.ipynb)**
    - Why sparse models scale better
    - Router mechanisms and load balancing
    - Implementing a simple MoE layer
    - Expert parallelism and challenges
    - Real-world MoE architectures

14. **[torch.compile & Performance Optimization](notebooks/03_advanced/14_torch_compile.ipynb)**
    - How torch.compile works (TorchInductor)
    - Just-in-time compilation benefits
    - Regional compilation for repeated modules
    - Mixed precision training (AMP)
    - Profiling and bottleneck identification

15. **[Production PyTorch: Best Practices](notebooks/03_advanced/15_production_pytorch.ipynb)**
    - Project structure and code organization
    - Model versioning and checkpointing
    - Quantization and model compression
    - ONNX export and deployment
    - Debugging and common pitfalls

## Prerequisites

- Basic Python programming knowledge
- Basic understanding of linear algebra and calculus (gradients)
- Familiarity with NumPy is helpful but not required

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Building_PyTorch_Learning_Platform
```

### 2. Create Environment with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies with uv

```bash
# Install PyTorch with CUDA (adjust index-url for your CUDA version)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
uv pip install -r requirements.txt
```

**Alternative: Using standard pip**
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and start with Topic 1!

## Dependencies

This course uses the latest PyTorch features (as of October 2025):

- **PyTorch 2.8+**: Latest features including FlexAttention, torch.compile improvements
- **torchvision**: For vision tasks and pretrained models
- **matplotlib & seaborn**: Visualizations
- **numpy & pandas**: Data manipulation
- **jupyter**: Interactive notebooks
- **tqdm**: Progress bars

See `requirements.txt` for complete list with pinned versions.

## Notebook Structure

Each notebook follows a consistent structure:

1. **Conceptual Overview**: Big picture understanding
2. **Why This Matters**: Motivation and real-world relevance
3. **Step-by-Step Implementation**: Detailed code with explanations
4. **Visualizations**: Diagrams and plots to aid understanding
5. **Mini Exercises**: Practice what you've learned
6. **Comprehensive Exercise**: End-of-notebook challenge
7. **Solutions**: Collapsible/hidden cells for self-study
8. **Further Reading**: Resources to deepen knowledge

## Learning Path

### For Complete Beginners
Start with Topic 1 and progress sequentially through all 15 topics.

### For Intermediate Users
If you're comfortable with PyTorch basics:
- Review Topics 1-5 quickly
- Focus on Topics 6-10 (especially 7-9 for Transformers)
- Deep dive into Topics 11-15

### For Advanced Users
If you want to learn specific techniques:
- **Attention & Transformers**: Topics 7-9
- **Optimization**: Topics 11, 14
- **Modern Architectures**: Topics 12-13
- **Production**: Topic 15

## Latest PyTorch Features Covered

This course incorporates cutting-edge PyTorch features (2025):

- **FlexAttention API**: Flexible attention implementations with automatic fusion
- **torch.compile**: JIT compilation with TorchInductor
- **Regional Compilation**: Efficient compilation for repeated modules (transformer layers)
- **cuDNN Fused Flash Attention**: 75% speedup on H100 GPUs
- **Control Flow Operators**: cond, while_loop for dynamic graphs
- **Scaled Dot Product Attention**: Native PyTorch attention function

## Tips for Success

1. **Code Along**: Don't just read - type and run every code example
2. **Experiment**: Modify parameters and see what happens
3. **Do the Exercises**: Theory alone isn't enough; practice solidifies understanding
4. **Visualize**: Use the provided visualizations and create your own
5. **Connect the Dots**: Each topic builds on previous ones - review when needed
6. **Ask Why**: If something isn't clear, go back and understand the motivation
7. **Build Projects**: After each level, try building something from scratch

## Common Pitfalls to Avoid

- **Skipping fundamentals**: Advanced topics assume solid basics
- **Not understanding shapes**: Tensor shape mismatches are the #1 bug
- **Ignoring computational graphs**: Understand how gradients flow
- **Copying code blindly**: Understand every line you write
- **Not checking device**: GPU vs CPU errors are common for beginners

## Project Ideas (After Completion)

After finishing the course, try these projects:

**Beginner Level:**
- MNIST digit classifier
- Fashion-MNIST with CNNs
- Linear regression on real-world data

**Intermediate Level:**
- Image classifier with custom dataset
- Sentiment analysis with attention
- Simple seq2seq translator

**Advanced Level:**
- Build a mini-GPT from scratch
- Implement a modern Vision Transformer
- Create a MoE-based classifier
- Fine-tune a pretrained model

## Contributing

Found an error or have a suggestion? Please open an issue or submit a pull request!

## Resources & References

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch 2.8 Release Notes](https://pytorch.org/blog/pytorch-2-8/)
- [Attention Is All You Need (Paper)](https://arxiv.org/abs/1706.03762)
- [Flash Attention (Paper)](https://arxiv.org/abs/2205.14135)
- [RoFormer: RoPE (Paper)](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models (Paper)](https://arxiv.org/abs/2305.13245)

## License

This course is released under the MIT License. Feel free to use, modify, and share!

## Acknowledgments

This course incorporates the latest research and best practices from the PyTorch community, academic papers, and production ML systems. Special thanks to the PyTorch team for their excellent documentation and the research community for advancing the field.

---

**Ready to master PyTorch? Start with [Topic 1: PyTorch Fundamentals](notebooks/01_beginner/01_pytorch_fundamentals.ipynb)!**
