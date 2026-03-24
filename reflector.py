from llm import generate_response


class Reflector:
    def reflect(self, agent_name: str, current_belief: str, memories: list[str]) -> str:
        memory_block = "\n".join(f"- {m}" for m in memories)

        prompt = f"""
You are {agent_name}.

Current belief:
{current_belief}

Recent discussion memories:
{memory_block}

Write one short reflection sentence from {agent_name}'s perspective.

Requirements:
- Under 20 words.
- Mention one tension, takeaway, or shift in thinking.
- Sound like an internal thought, not a formal summary.
"""

        return generate_response(prompt, temperature=0.6)