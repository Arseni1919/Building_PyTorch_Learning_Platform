# PyTorch Learning Platform - Project Instructions

## Project Overview
This is a comprehensive PyTorch learning platform built with Gradio for deployment on HuggingFace Spaces. The platform focuses on teaching PyTorch from basics to advanced concepts, with a strong emphasis on Transformer architectures and modern LLM building blocks.

## Core Requirements

### Platform Features
- **Topic-based learning**: Separate pages for each topic
- **4 difficulty levels**: Basic, Intermediate, Advanced, Professional
- **Interactive quizzes**: Multiple-choice (instant feedback) and open-ended (show answer on submit)
- **Progress tracking**: Session-based tracking of opened pages and completion status
- **Example projects**: End-of-section projects with hidden solutions (CPU-only)
- **Simple language**: Clear explanations with "why" for every concept
- **Conceptual connections**: Link concepts to overall PyTorch ecosystem
- **Markdown + code snippets**: Use Gradio markdown with syntax highlighting
- **Emojis**: Used strategically but not excessively

### Technical Requirements
- **Framework**: Gradio (for HuggingFace Spaces deployment)
- **Deployment target**: HuggingFace Spaces
- **Hardware**: CPU-only (no GPU required)
- **Best practices**: PyTorch project structure, modern conventions
- **Testing**: All code and links verified

## Curriculum Structure (29 Topics Total)

### BASIC LEVEL (6 topics) 📚
1. Introduction to PyTorch & Tensors
2. Autograd & Backpropagation
3. Building Neural Networks (nn.Module)
4. Loss Functions & Optimizers
5. Training Your First Model
6. Evaluation & Metrics

**Project**: Simple Image Classifier (Fashion-MNIST)

### INTERMEDIATE LEVEL (7 topics) 🧠
7. Custom Datasets & DataLoaders
8. Convolutional Neural Networks (CNNs)
9. Batch Normalization & Dropout
10. Transfer Learning
11. Advanced Optimizers & Schedulers
12. Model Saving & Checkpointing
13. Introduction to Embeddings

**Project**: Custom Dataset Trainer with augmentation

### ADVANCED LEVEL (10 topics) ⚡ **TRANSFORMER-FOCUSED**
14. Attention Mechanism from Scratch
15. Multi-Head Attention
16. Positional Encoding
17. RoPE (Rotary Position Embeddings)
18. The Transformer Architecture
19. Modern Transformer Components (RMSNorm, SwiGLU)
20. Grouped Query Attention (GQA)
21. Flash Attention
22. KV Cache & Efficient Inference
23. Mixture of Experts (MoE)

**Project**: Build a Mini-Transformer (encoder-only)

### PROFESSIONAL LEVEL (6 topics) 🚀
24. Production-Ready Training Loop
25. Model Optimization & Quantization
26. Distributed Training Concepts
27. PyTorch Best Practices
28. Debugging & Profiling
29. Deployment Strategies

**Project**: Mini Training Pipeline with configs

## Implementation Plan

### Phase 1: Project Setup & Structure
- [x] Create CLAUDE.md
- [ ] Create directory structure:
  ```
  Building_PyTorch_Learning_Platform/
  ├── app.py                          # Main Gradio application
  ├── requirements.txt                # Dependencies
  ├── README.md                       # HF Spaces card
  ├── .gitignore                      # Git exclusions
  ├── CLAUDE.md                       # This file
  ├── topics/
  │   ├── basic/                      # 6 topic files
  │   ├── intermediate/               # 7 topic files
  │   ├── advanced/                   # 10 topic files
  │   └── professional/               # 6 topic files
  ├── projects/                       # 4 example projects
  └── utils/
      ├── progress_tracker.py         # Progress tracking
      └── quiz_handler.py             # Quiz logic
  ```

### Phase 2: Core Utilities
- [ ] `utils/progress_tracker.py`: Session-based progress tracking with Gradio State
- [ ] `utils/quiz_handler.py`: Quiz rendering and validation logic

### Phase 3: Content Development (29 Topics)
Each topic file should include:
- Markdown content with clear explanations
- Code snippets with syntax highlighting
- "Why" explanations for every concept
- Connections to other topics and overall PyTorch ecosystem
- 2-3 quiz questions (mix of multiple-choice and open-ended)

**Content Guidelines**:
- Use simple, accessible language
- Explain WHY each component was created
- Show how concepts connect to transformer/LLM development
- Include runnable code examples (CPU-only)
- Add strategic emojis for visual appeal

### Phase 4: Example Projects
Create 4 hands-on projects with complete solutions:
1. **Basic**: Fashion-MNIST image classifier
2. **Intermediate**: Custom dataset trainer with augmentation
3. **Advanced**: Mini-transformer for text classification
4. **Professional**: Complete training pipeline

Each project should:
- Be achievable with learned concepts
- Run on CPU only
- Have hidden solution (revealed on button click)
- Include detailed explanations

### Phase 5: Gradio UI Implementation
Build `app.py` using Gradio Blocks:
- Home page with 4-level navigation
- Progress dashboard with completion percentage
- Topic pages with content + quizzes
- Project pages with hidden solutions
- Responsive layout with tabs/accordions

### Phase 6: HuggingFace Spaces Deployment
- [ ] `requirements.txt` with pinned versions (CPU-only PyTorch)
- [ ] `README.md` with proper YAML metadata for HF Spaces
- [ ] `.gitignore` for Python/venv files
- [ ] Verify all imports work without GPU

### Phase 7: Testing & Validation
- [ ] Test all navigation flows
- [ ] Verify all quizzes work correctly
- [ ] Test progress tracking across sessions
- [ ] Run every code example on CPU
- [ ] Test all 4 example projects
- [ ] Check markdown rendering
- [ ] Validate all links and references
- [ ] Verify conceptual coherence (topics build on each other)

## Key Design Principles

### Educational Philosophy
1. **Why before How**: Always explain why a concept exists before explaining how to implement it
2. **Conceptual Connections**: Link every topic to broader PyTorch/ML ecosystem
3. **Progressive Complexity**: Each level builds on previous knowledge
4. **Hands-on Learning**: Theory + Code + Projects
5. **Modern Relevance**: Focus on techniques used in current LLMs (2025)

### Transformer-Centric Approach
The curriculum is designed to gradually build towards understanding modern transformer architectures:
- Basic: PyTorch fundamentals
- Intermediate: Neural networks and practical ML
- Advanced: Deep dive into transformers and modern LLM components
- Professional: Production deployment and optimization

Key transformer topics include:
- Self-attention mechanism (from scratch)
- Multi-head attention
- Positional encodings (sinusoidal → RoPE)
- Modern components (RMSNorm, SwiGLU, GQA)
- Efficiency techniques (Flash Attention, KV cache)
- Scaling approaches (MoE)

### Technical Best Practices (to highlight in content)
1. **Project Structure**: Modular organization, separation of concerns
2. **Configuration Management**: YAML configs, Hydra/OmegaConf
3. **Reproducibility**: Seed setting, deterministic operations
4. **Code Quality**: Documentation, type hints, clear naming
5. **Testing**: Unit tests, integration tests
6. **Version Control**: Git best practices

## HuggingFace Spaces Requirements

### Essential Files
- `app.py`: Main Gradio application
- `requirements.txt`: All dependencies with versions

### Metadata for README.md
```yaml
---
title: PyTorch Learning Platform
emoji: 🔥
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---
```

### Hardware
- CPU Basic (2 vCPUs, 16 GB RAM, FREE)
- No GPU dependencies

### Python Version
- Python 3.9+ recommended

## Dependencies (for requirements.txt)
```
gradio>=5.0.0
torch>=2.2.0  # CPU version, includes Flash Attention support
torchvision>=0.17.0
numpy>=1.24.0
matplotlib>=3.7.0
```

## Progress Tracking Implementation
Use Gradio State to store:
- List of visited topic IDs
- Completion status per topic
- Quiz scores (optional)
- Current level

## Quiz System Implementation
### Multiple Choice
- Show all options as buttons/radio
- On selection, immediately show ✅ or ❌
- Display explanation for correct answer

### Open-Ended
- Text input area
- "Submit" button
- On submit, show model answer below
- Option to compare user answer

## Emoji Usage Guidelines
Use emojis strategically:
- Level indicators: 📚 (Basic), 🧠 (Intermediate), ⚡ (Advanced), 🚀 (Professional)
- Feedback: ✅ (correct), ❌ (incorrect)
- Topics: 🎯 (attention), 🔄 (training), 💾 (saving), etc.
- Don't overuse - keep it professional

## Continuation Instructions

If interrupted, continue from the current phase:

1. **Check CLAUDE.md** for overall plan
2. **Check todo list** (use TodoWrite tool) for current status
3. **Review completed files** to understand context
4. **Continue with next pending task** in the implementation plan
5. **Update todo list** as you complete tasks
6. **Test incrementally** as you build

### Current Status
- Project initialized
- CLAUDE.md created
- Next: Create directory structure and core files

## Testing Checklist

Before deployment to HuggingFace Spaces:

- [ ] All 29 topics render correctly
- [ ] All code snippets have syntax highlighting
- [ ] All quizzes function properly
- [ ] Progress tracking works across sessions
- [ ] All 4 example projects run on CPU
- [ ] Navigation works smoothly
- [ ] Markdown formatting is correct
- [ ] No broken links or imports
- [ ] PyTorch code examples are verified
- [ ] Best practices are clearly highlighted
- [ ] Transformer topics form coherent progression
- [ ] "Why" explanations are clear and accurate
- [ ] Concepts are properly connected

## Research Sources Used

1. **Gradio + HuggingFace Spaces**: Official docs, deployment guides
2. **PyTorch Best Practices**: Structure guidelines, style guides
3. **Transformer Architecture**: Attention mechanism, modern variants
4. **Modern LLM Components**: RoPE, GQA, MoE, Flash Attention, KV cache
5. **2025 LLM Architectures**: LLaMA 4, GPT-4, Claude 4, DeepSeek-V3

## Notes

- All code must run on CPU (no CUDA requirements)
- Platform targets learners from beginners to advanced
- Emphasis on understanding "why" not just "how"
- Transformer architecture is the crown jewel of the curriculum
- Modern LLM building blocks make content relevant to 2025
- Simple language throughout, avoid unnecessary jargon
- Connect concepts to show the bigger picture

---

**Last Updated**: 2025-10-14
**Status**: In Development
**Next Step**: Create directory structure and requirements.txt
