import ollama

def generate_response(prompt: str, temperature: float = 0.7) -> str:
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "system",
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
