import ollama
import numpy as np

# This file handles all communication with the local AI model (gemma3:4b via Ollama).
# Every time an agent needs to think, speak, or classify a stance, it calls generate_response().
# It also provides get_embedding() for semantic memory retrieval and belief drift scoring.
# Requires: ollama pull nomic-embed-text

def generate_response(prompt: str, temperature: float = 0.7) -> str:
    # Send a prompt to the local LLM and get a text response back.
    # temperature controls randomness: 0.0 = very consistent, 1.0 = more creative/varied
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
                # System message sets the overall behavior of the model
                "content": (
                    "You are a conversational agent in a multi-agent discussion. "
                    "Respond naturally and coherently. "
                    "Stay consistent with the assigned persona and current belief. "
                    "When asked to reason, be concise and specific. "
                    "Do not repeat instructions."
                )
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": temperature}
    )
    return response["message"]["content"].strip()


def get_embedding(text: str) -> list[float]:
    # Returns a vector representation of the text using a dedicated embedding model.
    # Falls back to an empty list if the model isn't available.
    try:
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    except Exception:
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Measures how similar two embedding vectors are. Returns 1.0 = identical, 0.0 = unrelated.
    if not a or not b:
        return 0.0
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
