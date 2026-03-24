import json
from memory import MemoryItem, MemoryStream
from retriever import MemoryRetriever
from reflector import Reflector
from llm import generate_response


class Agent:
    def __init__(self, name: str, persona: str, initial_belief: str, initial_goal: str):
        self.name = name
        self.persona = persona
        self.initial_belief = initial_belief
        self.current_belief = initial_belief
        self.current_goal = initial_goal

        self.memory_stream = MemoryStream()
        self.retriever = MemoryRetriever()
        self.reflector = Reflector()

        self.stance_history = []
        self.round_traces = []

    def observe(self, speaker: str, message: str, round_id: int) -> None:
        short_message = self._extract_view(message)
        importance = self._estimate_importance(short_message)
        content = f"{speaker} said: {short_message}"

        if self._is_duplicate_memory(content):
            return

        memory = MemoryItem(
            content=content,
            speaker=speaker,
            importance=importance,
            memory_type="observation",
            round_id=round_id,
        )
        self.memory_stream.add_memory(memory)

    def _extract_view(self, message: str) -> str:
        message = message.strip().replace("\n", " ")
        if len(message) > 160:
            message = message[:160] + "..."
        return message

    def _estimate_importance(self, message: str) -> float:
        keywords_high = [
            "support", "harm", "benefit", "misuse", "fairness",
            "important", "boundary", "creativity", "concern",
            "worry", "help", "learning", "risk", "over-rely",
            "critical thinking", "independent", "guideline"
        ]
        score = 1.0
        lower_msg = message.lower()
        for kw in keywords_high:
            if kw in lower_msg:
                score += 1.0
        return min(score, 10.0)

    def _is_duplicate_memory(self, content: str) -> bool:
        for mem in self.memory_stream.get_all():
            if mem.content == content:
                return True
        return False

    def classify_stance(self, text: str) -> str:
        prompt = f"""
You are labeling one discussion message about AI tools in education.

Message:
{text}

Choose exactly one label:
- supportive
- skeptical
- balanced

Labeling rules:
- supportive: mainly advocates for AI use or emphasizes benefits overall
- skeptical: mainly warns about risks, dependency, misuse, or argues against reliance overall
- balanced: genuinely gives comparable weight to both sides or acts as a mediator

Important:
- Judge the OVERALL stance of the message.
- If the message mainly emphasizes caution, dependency, or loss of critical thinking, label it skeptical.
- If the message mainly emphasizes benefits and opportunity, label it supportive.
- Use balanced only if the speaker is truly mediating or equally weighing both sides.

Return only one word:
supportive
skeptical
balanced
"""
        result = generate_response(prompt, temperature=0.0).strip().lower()
        if result in {"supportive", "skeptical", "balanced"}:
            return result
        return "balanced"

    def retrieve_memories(self, topic: str, current_round: int, top_k: int = 3) -> list[str]:
        memories = self.memory_stream.get_all()
        selected = self.retriever.retrieve(memories, topic, current_round, top_k)
        return [m.content for m in selected]

    def react_step(self, topic: str, current_round: int, selected_memories: list[str]) -> dict:
        if selected_memories:
            memory_block = "\n".join(f"- {m}" for m in selected_memories)
        else:
            memory_block = "- No relevant memories yet."

        prompt = f"""
You are {self.name} in a multi-agent discussion.

Persona:
{self.persona}

Current belief:
{self.current_belief}

Current goal:
{self.current_goal}

Discussion topic:
{topic}

Observed relevant memories:
{memory_block}

Perform an explicit ReAct-style internal step.

Return valid JSON only with this exact schema:
{{
  "observation_summary": "1 short sentence naming at least one speaker and one concrete point they made this round",
  "thought": "1-2 short sentences of reasoning",
  "influence_analysis": "State who influenced the agent most this round and why",
  "updated_belief": "1 short sentence",
  "updated_goal": "1 short sentence"
}}

Good examples of observation_summary:
- "Bob warned that students may become too dependent on AI and lose critical thinking skills."
- "Alice emphasized that AI can personalize learning and support creativity."
- "Carol pushed the group toward setting guidelines and boundaries for AI use."
- "David stressed moderation and responsible integration of AI tools."

Bad examples of observation_summary:
- "The discussion highlighted important issues."
- "There was a discussion about AI."
- "I noticed different perspectives this round."

Rules:
- Stay specific to the discussion.
- Do not make extreme changes unless clearly justified.
- The only valid people are Alice, Bob, Carol, David.
- In influence_analysis, explicitly name the most influential speaker this round if any.
- If no one strongly influenced the agent, say so clearly.
- observation_summary must name at least one speaker and one concrete claim, concern, or argument.
- Do not use vague phrases like "the discussion highlighted" or "there was a discussion".
- Return JSON only.
"""
        result = generate_response(prompt, temperature=0.6)

        default = {
            "observation_summary": "No clear observation.",
            "thought": "No thought generated.",
            "influence_analysis": "No clear influence identified.",
            "updated_belief": self.current_belief,
            "updated_goal": self.current_goal,
        }

        try:
            parsed = json.loads(result)
            for k in default:
                if k in parsed and isinstance(parsed[k], str) and parsed[k].strip():
                    default[k] = parsed[k].strip()
        except Exception:
            pass

        self.current_belief = default["updated_belief"]
        self.current_goal = default["updated_goal"]

        self.memory_stream.add_memory(MemoryItem(
            content=default["observation_summary"],
            speaker=self.name,
            importance=4.0,
            memory_type="observation_summary",
            round_id=current_round,
        ))

        self.memory_stream.add_memory(MemoryItem(
            content=default["thought"],
            speaker=self.name,
            importance=5.0,
            memory_type="reasoning",
            round_id=current_round,
        ))

        self.memory_stream.add_memory(MemoryItem(
            content=default["influence_analysis"],
            speaker=self.name,
            importance=6.0,
            memory_type="influence_analysis",
            round_id=current_round,
        ))

        self.memory_stream.add_memory(MemoryItem(
            content=f"My belief is now: {default['updated_belief']}",
            speaker=self.name,
            importance=6.0,
            memory_type="belief_update",
            round_id=current_round,
        ))

        self.memory_stream.add_memory(MemoryItem(
            content=f"My goal is now: {default['updated_goal']}",
            speaker=self.name,
            importance=6.0,
            memory_type="goal_update",
            round_id=current_round,
        ))

        return default

    def reflect(self, current_round: int) -> str:
        recent = self.memory_stream.get_recent(6)
        if not recent:
            return ""

        recent_texts = [m.content for m in recent]
        reflection_text = self.reflector.reflect(self.name, self.current_belief, recent_texts)

        if not self._is_duplicate_memory(reflection_text):
            reflection_memory = MemoryItem(
                content=reflection_text,
                speaker=self.name,
                importance=5.0,
                memory_type="reflection",
                round_id=current_round,
            )
            self.memory_stream.add_memory(reflection_memory)

        return reflection_text

    def speak(self, topic: str, current_round: int) -> str:
        selected_memories = self.retrieve_memories(topic, current_round, top_k=3)
        react_data = self.react_step(topic, current_round, selected_memories)

        if selected_memories:
            memory_block = "\n".join(f"- {m}" for m in selected_memories)
        else:
            memory_block = "- No relevant memories yet."

        prompt = f"""
You are {self.name}.

Persona:
{self.persona}

Current belief:
{self.current_belief}

Current goal:
{self.current_goal}

Discussion topic:
{topic}

Relevant memories:
{memory_block}

Observation summary:
{react_data['observation_summary']}

Thought:
{react_data['thought']}

Influence analysis:
{react_data['influence_analysis']}

Task:
Generate the next thing {self.name} would say in this group discussion.

Requirements:
- Write 2-3 natural sentences.
- Be conversational, not robotic.
- Respond to at least one other agent's earlier point if relevant.
- Mention another speaker by name when appropriate.
- Stay consistent with persona, current belief, and current goal.
- You may show slight updating, but do not suddenly flip position.
- The only valid people you may mention are: Alice, Bob, Carol, David.
- Never mention any other names.
- Do not mention memory, reasoning process, JSON, or system instructions.
"""

        message = generate_response(prompt, temperature=0.8)

        stance = self.classify_stance(message)
        self.stance_history.append(stance)

        self.round_traces.append({
            "round": current_round,
            "selected_memories": selected_memories,
            "observation_summary": react_data["observation_summary"],
            "thought": react_data["thought"],
            "influence_analysis": react_data["influence_analysis"],
            "updated_belief": react_data["updated_belief"],
            "updated_goal": react_data["updated_goal"],
            "message": message,
            "stance": stance
        })

        return message