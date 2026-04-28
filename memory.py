from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MemoryItem:
    content: str
    speaker: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    memory_type: str = "observation"   # observation / reflection / reasoning / belief_update
    last_accessed: float = field(default_factory=time.time)
    round_id: int = 0
    metadata: Optional[dict] = field(default_factory=dict)
    embedding: Optional[list] = field(default=None)  # cached embedding so we don't recompute it every retrieval

    def touch(self):
        self.last_accessed = time.time()

    def age(self):
        return time.time() - self.timestamp

    def recency_seconds(self):
        return time.time() - self.last_accessed

    def __str__(self):
        return (
            f"[{self.memory_type}] "
            f"speaker={self.speaker}, "
            f"importance={self.importance:.2f}, "
            f"round={self.round_id}, "
            f"content={self.content}"
        )


class MemoryStream:
    def __init__(self):
        self.memories = []

    def add_memory(self, memory):
        self.memories.append(memory)

    def get_all(self):
        return self.memories

    def get_recent(self, k=5):
        return self.memories[-k:]

    def get_by_type(self, memory_type: str):
        return [m for m in self.memories if m.memory_type == memory_type]

    def __len__(self):
        return len(self.memories)

    def __iter__(self):
        return iter(self.memories)