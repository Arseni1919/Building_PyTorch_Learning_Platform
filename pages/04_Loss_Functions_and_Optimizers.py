"""
Topic 4: Loss Functions & Optimizers
"""

import streamlit as st
from topics.basic.topic_04_loss_optimizers import get_topic_content, QUESTIONS
from utils.quiz_handler import QuizHandler

# Page config
st.set_page_config(
    page_title="04 - Loss Functions & Optimizers",
    page_icon="🎯",
    layout="wide"
)

# Get topic data
topic_data = get_topic_content()

# Display content
st.markdown(topic_data['content'])

# Quiz section
if QUESTIONS:
    st.markdown("---")
    st.markdown("## 📝 Knowledge Check")
    st.markdown("Test your understanding with these questions:")

    for idx, question in enumerate(QUESTIONS):
        st.markdown(f"### Question {idx + 1}")
        st.markdown(f"**{question.question_text}**")

        if question.question_type.value == "multiple_choice":
            # Multiple choice using radio buttons
            user_answer = st.radio(
                "Select your answer:",
                options=question.options,
                key=f"q{idx}",
                index=None
            )

            if st.button(f"Check Answer {idx + 1}", key=f"btn{idx}"):
                if user_answer:
                    is_correct, feedback = QuizHandler.check_answer(question, user_answer)
                    if is_correct:
                        st.success(feedback)
                    else:
                        st.error(feedback)
                else:
                    st.warning("Please select an answer first!")

        else:
            # Open-ended question
            user_answer = st.text_area(
                "Your answer:",
                key=f"q{idx}",
                height=100
            )

            if st.button(f"Submit Answer {idx + 1}", key=f"btn{idx}"):
                if user_answer.strip():
                    is_correct, feedback = QuizHandler.check_answer(question, user_answer)
                    st.info(feedback)
                else:
                    st.warning("Please write your answer first!")

        st.markdown("---")

# Navigation
col1, col2 = st.columns(2)
with col1:
    st.info("⬅️ Previous: **03 - Building Neural Networks**")
with col2:
    st.info("➡️ Next: **05 - Training Your First Model** (Coming Soon)")
