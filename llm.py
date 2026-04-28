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


def generate_consensus_summary(topic: str, agent_summaries: dict) -> str:
    # Synthesizes what happened across all trials into a presentation-ready paragraph.
    # agent_summaries: {name: {initial_belief, final_beliefs: [], stance_histories: []}}
    block = ""
    for name, data in agent_summaries.items():
        trajectories = "; ".join(
            " → ".join(h) for h in data["stance_histories"]
        )
        finals = " | ".join(data["final_beliefs"])
        block += (
            f"\n{name}:\n"
            f"  Initial belief: {data['initial_belief']}\n"
            f"  Final beliefs across trials: {finals}\n"
            f"  Stance trajectories: {trajectories}\n"
        )

    num_trials = len(next(iter(agent_summaries.values()))["final_beliefs"])
    prompt = f"""
You are summarizing a multi-agent debate on the topic: "{topic}"

The debate ran for {num_trials} independent trials. Here is how each agent evolved:
{block}

Write a 4-5 sentence consensus summary that:
1. Identifies the main point(s) of agreement that emerged across trials
2. Identifies the main point(s) of persistent disagreement
3. Notes which agent(s) shifted the most and in what direction
4. Ends with a one-sentence synthesis of the overall group conclusion

Be specific and insightful. Do not use bullet points or headers.
"""
    return generate_response(prompt, temperature=0.4)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Measures how similar two embedding vectors are. Returns 1.0 = identical, 0.0 = unrelated.
    if not a or not b:
        return 0.0
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
