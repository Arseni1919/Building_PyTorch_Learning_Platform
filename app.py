"""
PyTorch Learning Platform - Home Page (Streamlit)
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="PyTorch Learning Platform",
    page_icon="🔥",
    layout="wide"
)

# Title
st.title("🔥 PyTorch Learning Platform")
st.markdown("### Master PyTorch from Basics to Advanced Transformers")

st.markdown("---")

# Introduction
st.markdown("""
Master PyTorch from basics to advanced transformer architectures. This platform focuses on **understanding WHY** each concept exists and **HOW** it connects to modern LLMs.

**Use the sidebar** → to navigate to any topic!
""")

st.markdown("---")

# Basic Level
st.markdown("## 📚 Basic Level - Foundation Concepts")
st.markdown("""
1. **Introduction to PyTorch & Tensors** 🔢
2. **Autograd & Backpropagation** 🔄
3. **Building Neural Networks (nn.Module)** 🧠
4. **Loss Functions & Optimizers** 🎯
5. **Training Your First Model** 🚀
6. **Evaluation & Metrics** ✅
""")

st.markdown("---")

# Intermediate Level
st.markdown("## 🧠 Intermediate Level - Practical Deep Learning")
st.markdown("""
7. Custom Datasets & DataLoaders 📂
8. Convolutional Neural Networks (CNNs) 🖼️
9. Batch Normalization & Dropout 🔧
10. Transfer Learning 🔄
11. Advanced Optimizers & Schedulers 📈
12. Model Saving & Checkpointing 💾
13. Introduction to Embeddings 🔤
""")

st.markdown("---")

# Advanced Level
st.markdown("## ⚡ Advanced Level - Transformer Architectures & Modern LLMs")
st.markdown("**The heart of modern AI! Deep dive into transformers:**")
st.markdown("""
14. Attention Mechanism from Scratch 🎯
15. Multi-Head Attention 🧩
16. Positional Encoding 📍
17. RoPE (Rotary Position Embeddings) 🔄
18. The Transformer Architecture 🏗️
19. Modern Transformer Components (RMSNorm, SwiGLU) ⚡
20. Grouped Query Attention (GQA) 🔍
21. Flash Attention ⚡
22. KV Cache & Efficient Inference 💨
23. Mixture of Experts (MoE) 🎓
""")

st.markdown("---")

# Professional Level
st.markdown("## 🚀 Professional Level - Production & Optimization")
st.markdown("""
24. Production-Ready Training Loop 🔄
25. Model Optimization & Quantization 📊
26. Distributed Training Concepts 🌐
27. PyTorch Best Practices 📋
28. Debugging & Profiling 🔍
29. Deployment Strategies 🚀
""")

st.markdown("---")

# What makes it special
st.markdown("## 💡 What Makes This Platform Special?")
st.markdown("""
- **Transformer-Focused**: 10 dedicated topics on attention mechanisms and modern LLM architectures
- **Conceptual Connections**: Understand how concepts relate to each other
- **Hands-On Projects**: Build real projects at the end of each level
- **CPU-Friendly**: All examples run on CPU (no GPU needed!)
- **Simple Language**: Clear explanations of complex topics
- **Interactive Quizzes**: Test your understanding with each topic
""")

st.markdown("---")

# Projects
st.markdown("## 🎯 Projects")
st.markdown("""
After completing each level, build a hands-on project:

- **📚 Basic**: Fashion-MNIST Classifier
- **🧠 Intermediate**: Custom Dataset Trainer with Augmentation
- **⚡ Advanced**: Mini-Transformer for Text Classification
- **🚀 Professional**: Production Training Pipeline
""")

st.markdown("---")

# Get started
st.info("👈 **Select a topic from the sidebar to begin your learning journey!**")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit** | Focus on Transformers & Modern LLM Architectures")
