"""
PyTorch Learning Platform - Single Page App with Dynamic Content
Uses buttons for navigation instead of links
"""

import gradio as gr
from topics.basic.topic_01_pytorch_tensors import get_topic_content as get_topic_01, QUESTIONS as questions_01
from topics.basic.topic_02_autograd import get_topic_content as get_topic_02, QUESTIONS as questions_02
from utils.quiz_handler import QuizHandler

# Topic metadata for navigation
TOPICS = [
    ("01", "Introduction to PyTorch & Tensors", get_topic_01, questions_01, None),
    ("02", "Autograd & Backpropagation", get_topic_02, questions_02, "01"),
    # More topics will be added here
]

def create_home_content():
    """Generate home page content."""
    return """# 🔥 Welcome to the PyTorch Learning Platform!

Master PyTorch from basics to advanced transformer architectures. This platform focuses on **understanding WHY** each concept exists and **HOW** it connects to modern LLMs.

---

## 📚 Basic Level - Foundation Concepts

**Click the navigation buttons above to access any topic:**

1. Introduction to PyTorch & Tensors 🔢
2. Autograd & Backpropagation 🔄
3. Building Neural Networks (nn.Module) 🧠
4. Loss Functions & Optimizers 🎯
5. Training Your First Model 🚀
6. Evaluation & Metrics ✅

---

## 🧠 Intermediate Level - Practical Deep Learning

7. Custom Datasets & DataLoaders 📂
8. Convolutional Neural Networks (CNNs) 🖼️
9. Batch Normalization & Dropout 🔧
10. Transfer Learning 🔄
11. Advanced Optimizers & Schedulers 📈
12. Model Saving & Checkpointing 💾
13. Introduction to Embeddings 🔤

---

## ⚡ Advanced Level - Transformer Architectures & Modern LLMs

**The heart of modern AI! Deep dive into transformers:**

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

---

## 🚀 Professional Level - Production & Optimization

24. Production-Ready Training Loop 🔄
25. Model Optimization & Quantization 📊
26. Distributed Training Concepts 🌐
27. PyTorch Best Practices 📋
28. Debugging & Profiling 🔍
29. Deployment Strategies 🚀

---

## 💡 What Makes This Platform Special?

- **Transformer-Focused**: 10 dedicated topics on attention mechanisms and modern LLM architectures
- **Conceptual Connections**: Understand how concepts relate to each other
- **Hands-On Projects**: Build real projects at the end of each level
- **CPU-Friendly**: All examples run on CPU (no GPU needed!)
- **Simple Language**: Clear explanations of complex topics
- **Interactive Quizzes**: Test your understanding with each topic

---

## 🎯 Projects

After completing each level, build a hands-on project:

- **📚 Basic**: Fashion-MNIST Classifier
- **🧠 Intermediate**: Custom Dataset Trainer with Augmentation
- **⚡ Advanced**: Mini-Transformer for Text Classification
- **🚀 Professional**: Production Training Pipeline

---

## 🚀 Get Started!

Use the navigation buttons above to jump to any topic. Topics currently available: 1, 2. More coming soon!
"""

def create_topic_page(topic_id):
    """Create content for a specific topic."""
    for tid, title, get_content, questions, prev_id in TOPICS:
        if tid == topic_id:
            topic_data = get_content()
            content = topic_data['content']

            # Add navigation info
            nav_info = f"\n\n---\n\n**Topic {tid}**: {title}\n\n"
            if prev_id:
                nav_info += f"⬅️ Previous topic available | "
            nav_info += "🏠 Home | ➡️ Next topic available\n\n---\n\n"

            return nav_info + content, questions, title

    return create_home_content(), [], "Home"

def render_quiz(questions):
    """Render quiz questions as HTML."""
    if not questions:
        return ""

    quiz_html = "\n\n---\n\n## 📝 Knowledge Check\n\n"
    for idx, q in enumerate(questions):
        quiz_html += f"\n### Question {idx + 1}\n\n**{q.question_text}**\n\n"
        if q.question_type.value == "multiple_choice":
            for opt in q.options:
                quiz_html += f"- {opt}\n"

    return quiz_html

def main():
    """Create the main Gradio interface."""

    with gr.Blocks(title="🔥 PyTorch Learning Platform", theme=gr.themes.Soft()) as app:
        # State to track current page
        current_topic = gr.State("home")

        # Header
        gr.Markdown("# 🔥 PyTorch Learning Platform")
        gr.Markdown("### Master PyTorch from Basics to Advanced Transformers")

        # Navigation bar
        with gr.Row():
            home_btn = gr.Button("🏠 Home", size="sm")
            topic_01_btn = gr.Button("01. PyTorch & Tensors", size="sm")
            topic_02_btn = gr.Button("02. Autograd", size="sm")

        gr.Markdown("---")

        # Main content area
        content_area = gr.Markdown(value=create_home_content())

        # Quiz area (initially hidden)
        with gr.Column(visible=False) as quiz_section:
            gr.Markdown("## 📝 Knowledge Check")
            quiz_questions = gr.Column()

        # Navigation functions
        def go_home():
            return create_home_content(), "home", gr.Column(visible=False)

        def go_to_topic(topic_id):
            content, questions, title = create_topic_page(topic_id)
            quiz_html = render_quiz(questions)
            full_content = content + quiz_html

            # Show quiz section if there are questions
            show_quiz = len(questions) > 0
            return full_content, topic_id, gr.Column(visible=show_quiz)

        # Wire up navigation
        home_btn.click(
            go_home,
            outputs=[content_area, current_topic, quiz_section]
        )

        topic_01_btn.click(
            lambda: go_to_topic("01"),
            outputs=[content_area, current_topic, quiz_section]
        )

        topic_02_btn.click(
            lambda: go_to_topic("02"),
            outputs=[content_area, current_topic, quiz_section]
        )

        # Footer
        gr.Markdown("""
        ---
        **Built with ❤️ using Gradio** | Focus on Transformers & Modern LLM Architectures
        """)

    return app

if __name__ == "__main__":
    app = main()
    app.launch()
