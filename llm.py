from openai import OpenAI

client = OpenAI()

def generate_response(prompt: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
        temperature=temperature
    )
    return response.choices[0].message.content.strip()