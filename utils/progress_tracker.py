"""
Progress tracking utility for the PyTorch Learning Platform.
Manages user progress through topics using Gradio State.
"""

from typing import Dict, List, Set


class ProgressTracker:
    """Tracks user progress through the learning platform."""

    def __init__(self):
        """Initialize a new progress tracker."""
        self.visited_topics: Set[str] = set()
        self.completed_topics: Set[str] = set()
        self.quiz_scores: Dict[str, float] = {}

    def mark_visited(self, topic_id: str) -> None:
        """Mark a topic as visited.

        Args:
            topic_id: Unique identifier for the topic
        """
        self.visited_topics.add(topic_id)

    def mark_completed(self, topic_id: str) -> None:
        """Mark a topic as completed (including quiz).

        Args:
            topic_id: Unique identifier for the topic
        """
        self.completed_topics.add(topic_id)
        self.mark_visited(topic_id)

    def save_quiz_score(self, topic_id: str, score: float) -> None:
        """Save a quiz score for a topic.

        Args:
            topic_id: Unique identifier for the topic
            score: Quiz score (0.0 to 1.0)
        """
        self.quiz_scores[topic_id] = score

    def is_visited(self, topic_id: str) -> bool:
        """Check if a topic has been visited.

        Args:
            topic_id: Unique identifier for the topic

        Returns:
            True if visited, False otherwise
        """
        return topic_id in self.visited_topics

    def is_completed(self, topic_id: str) -> bool:
        """Check if a topic has been completed.

        Args:
            topic_id: Unique identifier for the topic

        Returns:
            True if completed, False otherwise
        """
        return topic_id in self.completed_topics

    def get_progress_stats(self, total_topics: int) -> Dict[str, any]:
        """Get overall progress statistics.

        Args:
            total_topics: Total number of topics in the platform

        Returns:
            Dictionary with progress statistics
        """
        visited_count = len(self.visited_topics)
        completed_count = len(self.completed_topics)

        return {
            'visited_count': visited_count,
            'completed_count': completed_count,
            'total_topics': total_topics,
            'visited_percentage': (visited_count / total_topics * 100) if total_topics > 0 else 0,
            'completed_percentage': (completed_count / total_topics * 100) if total_topics > 0 else 0,
            'quiz_scores': self.quiz_scores.copy()
        }

    def get_level_progress(self, level_topic_ids: List[str]) -> Dict[str, any]:
        """Get progress statistics for a specific level.

        Args:
            level_topic_ids: List of topic IDs in this level

        Returns:
            Dictionary with level progress statistics
        """
        total = len(level_topic_ids)
        visited = sum(1 for tid in level_topic_ids if self.is_visited(tid))
        completed = sum(1 for tid in level_topic_ids if self.is_completed(tid))

        return {
            'total': total,
            'visited': visited,
            'completed': completed,
            'visited_percentage': (visited / total * 100) if total > 0 else 0,
            'completed_percentage': (completed / total * 100) if total > 0 else 0
        }

    def to_dict(self) -> Dict:
        """Convert tracker state to dictionary for serialization.

        Returns:
            Dictionary representation of tracker state
        """
        return {
            'visited_topics': list(self.visited_topics),
            'completed_topics': list(self.completed_topics),
            'quiz_scores': self.quiz_scores.copy()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProgressTracker':
        """Create a ProgressTracker from a dictionary.

        Args:
            data: Dictionary containing tracker state

        Returns:
            New ProgressTracker instance
        """
        tracker = cls()
        tracker.visited_topics = set(data.get('visited_topics', []))
        tracker.completed_topics = set(data.get('completed_topics', []))
        tracker.quiz_scores = data.get('quiz_scores', {}).copy()
        return tracker


def create_progress_tracker() -> ProgressTracker:
    """Factory function to create a new progress tracker.

    Returns:
        New ProgressTracker instance
    """
    return ProgressTracker()
