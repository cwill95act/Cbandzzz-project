# Architecture

This document describes how the Cbandzzz multi-agent debate simulator is structured internally.

---

## Module Overview

### `main.py` — Simulation Runner

The entry point for the project. Contains:

- `build_agents()` — instantiates the four agents with their personas, beliefs, and goals.
- `run_one_simulation(trial_id, rounds)` — runs a single trial: agents take turns speaking for N rounds, then reflect.
- `save_trial_json()` / `save_trial_summary()` — write structured output after each trial.
- `run_experiments(num_trials, rounds)` — loops over trials, then calls the LLM for a consensus summary.

**To run:** `python main.py`

---

### `agent.py` — Agent Class

The core simulation object. Each agent has:

| Attribute | Description |
|-----------|-------------|
| `name` | Display name (Alice, Bob, Carol, David) |
| `persona` | Background description shaping how the agent reasons |
| `initial_belief` | Starting position, stored for drift calculation |
| `current_belief` | Evolves each round based on discussion |
| `current_goal` | Conversational objective, can shift over time |
| `memory_stream` | `MemoryStream` instance storing all observations and reflections |
| `retriever` | `MemoryRetriever` for pulling relevant memories |
| `reflector` | `Reflector` for generating post-round reflections |
| `stance_history` | List of labels: `supportive`, `skeptical`, or `balanced` |
| `round_traces` | Full log of every round (belief, goal, message, drift, etc.) |

**Key methods:**

- `speak(topic, current_round, previous_round_messages)` — retrieves memories, calls the LLM, updates belief/goal/stance, appends a trace.
- `observe(speaker_name, message, round_id)` — stores another agent's message in memory.
- `reflect(current_round)` — generates a high-level reflection and stores it.

---

### `memory.py` — Memory System

- `MemoryItem` — a single memory entry with text, embedding, round, and recency.
- `MemoryStream` — stores all `MemoryItem` objects for an agent; supports `add()` and `get_all()`.

---

### `retriever.py` — Memory Retrieval

- `MemoryRetriever` — given a query string, computes its embedding and returns the top-K memories from the stream using cosine similarity.

This allows each agent to recall the most contextually relevant past observations before formulating a response.

---

### `reflector.py` — Reflection Generator

- `Reflector` — after each round, calls the LLM to generate a short high-level reflection based on recent observations. The reflection is stored as a memory item.

---

### `llm.py` — LLM Interface

Wraps all direct calls to the OpenAI API:

- `generate_response(prompt)` — single-turn text completion.
- `get_embedding(text)` — returns a vector embedding for a string.
- `cosine_similarity(a, b)` — computes the cosine similarity between two vectors.
- `generate_consensus_summary(topic, agent_summaries)` — given per-agent final beliefs and stance histories, asks the LLM to write a synthesis paragraph.

---

### `analyze.py` — Post-Run Analysis

Reads the `trial_N.json` files produced by `main.py` and generates five visualizations:

| Output file | What it shows |
|-------------|---------------|
| `stance_evolution.png` | Line chart: each agent's stance label per round |
| `venn_diagram.png` | Keyword overlap across trials |
| `influence_network.png` | Directed graph: who influenced whom most |
| `belief_drift.png` | Cosine drift of each agent's belief from round 1 |
| `convergence_curve.png` | Average pairwise divergence in stance over rounds |

**To run:** `python analyze.py` (requires trial JSON files to exist)

---

### `build_presentation.py` — Presentation Builder

Assembles a PowerPoint (`debate_analysis_final.pptx`) from the PNG charts and summary text files.

---

## Data Flow

```
main.py
  |
  +-- build_agents()
  |     agent.py (Agent objects)
  |       memory.py (MemoryStream)
  |       retriever.py (MemoryRetriever)
  |       reflector.py (Reflector)
  |       llm.py (generate_response, get_embedding)
  |
  +-- run_experiments()
        |
        per trial:
          per round:
            agent.speak()  ->  llm.generate_response()
            agent.observe()
          agent.reflect()  ->  llm.generate_response()
        |
        save_trial_json()       ->  trial_N.json
        save_trial_summary()    ->  trial_N_summary.txt
        |
        generate_consensus_summary()  ->  consensus_summary.txt

analyze.py
  reads trial_N.json files
  produces PNG visualizations

build_presentation.py
  reads PNGs + summary TXT
  produces PPTX / PDF
```

---

## Agent Loop (per round)

```
for each agent:
  1. Retrieve top-K relevant memories (via cosine similarity on embeddings)
  2. Build a prompt: persona + current belief + goal + memories + prior round messages
  3. Call LLM -> get message text
  4. Extract stance label from message
  5. Update current_belief and current_goal from LLM response
  6. Compute belief_drift (cosine distance from initial_belief embedding)
  7. Append round_trace
  8. Broadcast message to all other agents (agent.observe())

after all agents speak:
  for each agent:
    9. Generate reflection from recent observations (Reflector)
    10. Store reflection as a memory item
```
