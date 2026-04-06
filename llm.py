import ollama

# This file handles all communication with the local AI model (gemma3:4b via Ollama).
# Every time an agent needs to think, speak, or classify a stance, it calls generate_response().

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
