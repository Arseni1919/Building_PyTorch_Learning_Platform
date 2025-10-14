"""
Simple test of Gradio multipage functionality with sidebar
"""

import gradio as gr

# Main page with sidebar
with gr.Blocks() as demo:
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## 📚 Navigation")
            gr.Markdown("---")
            gr.Markdown("**Basic Topics**")
            gr.Markdown("- Home\n- Topic 1\n- Topic 2")
            gr.Markdown("---")
            gr.Markdown("**Advanced Topics**")
            gr.Markdown("- Coming soon...")

        # Main content
        with gr.Column(scale=4):
            gr.Markdown("# 🏠 Home Page")
            gr.Markdown("Welcome to the PyTorch Learning Platform!")

            name = gr.Textbox(label="Your Name")
            output = gr.Textbox(label="Greeting")
            greet_btn = gr.Button("Greet Me")

            def greet(name):
                return f"Hello {name}! Welcome to PyTorch learning!"

            greet_btn.click(greet, inputs=name, outputs=output)

# Second page - Topic 1
with demo.route("Topic 1: PyTorch & Tensors", "/topic_01"):
    with gr.Row():
        # Sidebar (repeated for consistency)
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## 📚 Navigation")
            gr.Markdown("---")
            gr.Markdown("**Basic Topics**")
            gr.Markdown("- Home\n- **Topic 1** ←\n- Topic 2")
            gr.Markdown("---")
            gr.Markdown("**Advanced Topics**")
            gr.Markdown("- Coming soon...")

        # Main content
        with gr.Column(scale=4):
            gr.Markdown("# Topic 1: Introduction to PyTorch & Tensors 🔢")
            gr.Markdown("""
            ## What is PyTorch?

            PyTorch is an open-source machine learning framework.

            This is the first topic page!
            """)

            topic_input = gr.Textbox(label="What did you learn?")
            submit_btn = gr.Button("Submit")
            feedback = gr.Textbox(label="Feedback")

            def give_feedback(text):
                return f"Great! You learned: {text}"

            submit_btn.click(give_feedback, inputs=topic_input, outputs=feedback)

# Third page - Topic 2
with demo.route("Topic 2: Autograd", "/topic_02"):
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## 📚 Navigation")
            gr.Markdown("---")
            gr.Markdown("**Basic Topics**")
            gr.Markdown("- Home\n- Topic 1\n- **Topic 2** ←")
            gr.Markdown("---")
            gr.Markdown("**Advanced Topics**")
            gr.Markdown("- Coming soon...")

        # Main content
        with gr.Column(scale=4):
            gr.Markdown("# Topic 2: Autograd & Backpropagation 🔄")
            gr.Markdown("""
            ## What is Autograd?

            Autograd is PyTorch's automatic differentiation engine.

            This is the second topic page!
            """)

if __name__ == "__main__":
    demo.launch()
