import math
from memory import MemoryItem


class MemoryRetriever:
    def __init__(self, recency_weight=1.0, importance_weight=1.0, relevance_weight=1.2):
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight

    def _recency_score(self, memory: MemoryItem, current_round: int) -> float:
        distance = max(current_round - memory.round_id, 0)
        return math.exp(-0.5 * distance)

    def _importance_score(self, memory: MemoryItem) -> float:
        return memory.importance / 10.0

    def _relevance_score(self, memory: MemoryItem, topic: str) -> float:
        memory_words = set(memory.content.lower().split())
        topic_words = set(topic.lower().split())

        extra_keywords = {
            "ai", "education", "students", "learning", "fairness",
            "creativity", "misuse", "boundary", "support", "harm"
        }
        topic_words = topic_words | extra_keywords

        if not topic_words:
            return 0.0

        overlap = len(memory_words & topic_words)
        return overlap / len(topic_words)

    def score_memory(self, memory: MemoryItem, topic: str, current_round: int) -> float:
        recency = self._recency_score(memory, current_round)
        importance = self._importance_score(memory)
        relevance = self._relevance_score(memory, topic)

        return (
            self.recency_weight * recency
            + self.importance_weight * importance
            + self.relevance_weight * relevance
        )

    def retrieve(self, memories: list[MemoryItem], topic: str, current_round: int, top_k: int = 3) -> list[MemoryItem]:
        scored = []
        for memory in memories:
            score = self.score_memory(memory, topic, current_round)
            scored.append((score, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_memories = [m for _, m in scored[:top_k]]

        for memory in top_memories:
            memory.touch()

        return top_memories