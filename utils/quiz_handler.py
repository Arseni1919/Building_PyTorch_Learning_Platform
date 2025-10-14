"""
Quiz handler utility for the PyTorch Learning Platform.
Manages quiz questions, validation, and UI rendering.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of quiz questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"


@dataclass
class Question:
    """Represents a single quiz question."""
    question_text: str
    question_type: QuestionType
    correct_answer: str  # For MC: the correct option; for open: model answer
    options: Optional[List[str]] = None  # Only for multiple choice
    explanation: Optional[str] = None  # Explanation of the answer


class QuizHandler:
    """Handles quiz creation, validation, and rendering."""

    @staticmethod
    def create_multiple_choice(
        question: str,
        options: List[str],
        correct_answer: str,
        explanation: str = ""
    ) -> Question:
        """Create a multiple choice question.

        Args:
            question: The question text
            options: List of answer options
            correct_answer: The correct answer (must be in options)
            explanation: Explanation of why the answer is correct

        Returns:
            Question object

        Raises:
            ValueError: If correct_answer is not in options
        """
        if correct_answer not in options:
            raise ValueError(f"Correct answer '{correct_answer}' not found in options")

        return Question(
            question_text=question,
            question_type=QuestionType.MULTIPLE_CHOICE,
            correct_answer=correct_answer,
            options=options,
            explanation=explanation
        )

    @staticmethod
    def create_open_ended(
        question: str,
        model_answer: str,
        explanation: str = ""
    ) -> Question:
        """Create an open-ended question.

        Args:
            question: The question text
            model_answer: The model/expected answer
            explanation: Additional explanation

        Returns:
            Question object
        """
        return Question(
            question_text=question,
            question_type=QuestionType.OPEN_ENDED,
            correct_answer=model_answer,
            options=None,
            explanation=explanation
        )

    @staticmethod
    def check_answer(question: Question, user_answer: str) -> Tuple[bool, str]:
        """Check if a user's answer is correct.

        Args:
            question: The Question object
            user_answer: The user's submitted answer

        Returns:
            Tuple of (is_correct, feedback_message)
        """
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            is_correct = user_answer == question.correct_answer
            if is_correct:
                feedback = f"✅ Correct! {question.explanation}" if question.explanation else "✅ Correct!"
            else:
                feedback = f"❌ Incorrect. The correct answer is: {question.correct_answer}\n\n{question.explanation}" if question.explanation else f"❌ Incorrect. The correct answer is: {question.correct_answer}"
            return is_correct, feedback
        else:
            # For open-ended, we just show the model answer
            feedback = f"**Model Answer:**\n\n{question.correct_answer}"
            if question.explanation:
                feedback += f"\n\n**Explanation:**\n\n{question.explanation}"
            return True, feedback  # Open-ended are always "correct" for progress tracking

    @staticmethod
    def format_multiple_choice_options(options: List[str]) -> List[str]:
        """Format multiple choice options with labels (A, B, C, D).

        Args:
            options: List of option texts

        Returns:
            List of formatted options
        """
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        return [f"{labels[i]}. {opt}" for i, opt in enumerate(options[:len(labels)])]

    @staticmethod
    def calculate_quiz_score(questions: List[Question], user_answers: Dict[int, str]) -> float:
        """Calculate quiz score based on user answers.

        Args:
            questions: List of Question objects
            user_answers: Dictionary mapping question index to user answer

        Returns:
            Score as a float between 0.0 and 1.0
        """
        if not questions:
            return 0.0

        correct_count = 0
        for idx, question in enumerate(questions):
            if idx in user_answers:
                user_answer = user_answers[idx]
                if question.question_type == QuestionType.MULTIPLE_CHOICE:
                    if user_answer == question.correct_answer:
                        correct_count += 1
                else:
                    # Open-ended questions are counted as correct if answered
                    if user_answer and user_answer.strip():
                        correct_count += 1

        return correct_count / len(questions)


# Helper functions for creating common question patterns

def create_definition_question(
    concept: str,
    correct_definition: str,
    wrong_definitions: List[str],
    explanation: str = ""
) -> Question:
    """Create a 'What is X?' style multiple choice question.

    Args:
        concept: The concept being defined
        correct_definition: The correct definition
        wrong_definitions: List of incorrect definitions (should be 2-3)
        explanation: Additional explanation

    Returns:
        Question object
    """
    options = [correct_definition] + wrong_definitions
    # Simple shuffle to avoid correct answer always being first
    options = [options[i] for i in [1, 0, 2, 3][:len(options)]]

    return QuizHandler.create_multiple_choice(
        question=f"What is {concept}?",
        options=options,
        correct_answer=correct_definition,
        explanation=explanation
    )


def create_code_question(
    question: str,
    code_options: List[str],
    correct_code: str,
    explanation: str = ""
) -> Question:
    """Create a code-based multiple choice question.

    Args:
        question: The question text
        code_options: List of code snippet options
        correct_code: The correct code snippet
        explanation: Explanation of why the code is correct

    Returns:
        Question object
    """
    return QuizHandler.create_multiple_choice(
        question=question,
        options=code_options,
        correct_answer=correct_code,
        explanation=explanation
    )


def create_why_question(
    concept: str,
    model_answer: str,
    explanation: str = ""
) -> Question:
    """Create a 'Why is X important?' style open-ended question.

    Args:
        concept: The concept being asked about
        model_answer: The model answer explaining why
        explanation: Additional context

    Returns:
        Question object
    """
    return QuizHandler.create_open_ended(
        question=f"Why is {concept} important in PyTorch/deep learning?",
        model_answer=model_answer,
        explanation=explanation
    )
