# Cbandzzz-project — Multi-Agent LLM Debate Simulator

A research simulation where LLM-powered agents debate a topic, evolve their beliefs over multiple rounds, and generate consensus summaries. Built with OpenAI's API.

---

## What This Project Does

This project simulates a structured debate between four AI agents, each with a unique persona, starting belief, and conversational goal. Over several rounds the agents:

- **Speak** — each agent formulates a message based on its current beliefs, retrieved memories, and what others said in the previous round.
- **Observe** — agents listen to one another and store observations in a memory stream.
- **Reflect** — after each round, agents generate a high-level reflection and update their beliefs and goals.
- **Evolve** — stance histories and belief-drift scores are tracked across every round and trial.

After all trials complete, the simulation generates visualizations and an LLM-written consensus summary.

The default debate topic is: **"Whether students should use AI tools in education."**

---

## Project Structure

```
Cbandzzz-project/
│
├── main.py                   # Entry point — runs all trials and saves results
├── agent.py                  # Agent class (speak, observe, reflect, stance tracking)
├── llm.py                    # OpenAI API helpers (generate_response, embeddings, consensus)
├── memory.py                 # MemoryItem and MemoryStream classes
├── retriever.py              # MemoryRetriever — pulls relevant memories via cosine similarity
├── reflector.py              # Reflector — generates high-level reflections after each round
│
├── analyze.py                # Post-run analysis — loads trial JSON files, produces charts
├── build_presentation.py     # Builds a PowerPoint presentation from results
├── analysis.ipynb            # Jupyter notebook for interactive exploration
│
├── outputs/                  # Generated after running the simulation
│   ├── trial_N.json              # Full structured data for each trial
│   ├── trial_N_summary.txt       # Human-readable summary per trial
│   ├── overall_summary.txt       # Combined summary across all trials
│   ├── consensus_summary.txt     # LLM-generated consensus paragraph
│   ├── stance_evolution.png      # How each agent's stance shifted round by round
│   ├── belief_drift.png          # How far each agent's belief drifted from the start
│   ├── convergence_curve.png     # Did the group converge or stay divided?
│   ├── influence_network.png     # Who influenced whom most (directed graph)
│   ├── conversation_heatmap.png  # Topic frequency heatmap across agents and rounds
│   ├── venn_diagram.png          # Shared themes across trials
│   └── debate_analysis_final.pptx / presentation.pdf
│
├── .help/                    # Internal helper prompts / configuration
├── .gitignore
└── README.md
```

---

## Agents

| Agent | Persona | Starting Belief |
|-------|---------|-----------------|
| **Alice** | Optimistic about ed-tech, open to equity concerns | AI tools can support learning and creativity |
| **Bob** | Skeptical, acknowledges thoughtful use cases exist | AI carries real risks of dependency and misuse |
| **Carol** | Balanced mediator | AI tools are useful, but schools need clear boundaries |
| **David** | Mildly supportive but cautious | AI tools can help students only when used responsibly |

---

## Quickstart

### 1. Install dependencies

```bash
pip install openai numpy matplotlib networkx matplotlib-venn python-pptx
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="your-key-here"
```

### 3. Run the simulation

```bash
python main.py
```

This runs **5 trials x 5 rounds** by default and writes output files to the project root.

### 4. Generate visualizations

```bash
python analyze.py
```

Reads the `trial_N.json` files and produces five PNG charts.

### 5. Build the presentation (optional)

```bash
python build_presentation.py
```

Generates `debate_analysis_final.pptx` from the charts and summary text.

---

## How It Works

```
main.py
  build_agents()                      -- Creates 4 Agent objects
  run_one_simulation()
    agent.speak()                     -- Agent formulates a message
      retriever.retrieve()            -- Pull relevant memories via cosine similarity
      llm.generate_response()         -- Call OpenAI API
    agent.observe()                   -- Store others' messages in memory stream
    agent.reflect()                   -- Generate end-of-round reflection
  save_trial_json()                   -- Write trial_N.json
  save_trial_summary()                -- Write trial_N_summary.txt
  generate_consensus_summary()        -- LLM writes final synthesis
```

**Memory system:** Each agent has a `MemoryStream` that stores observations and reflections as `MemoryItem` objects. The `MemoryRetriever` uses OpenAI embeddings and cosine similarity to surface the most relevant memories when an agent is about to speak.

**Stance tracking:** Each round, an agent classifies its position as `supportive`, `skeptical`, or `balanced`. These labels form a `stance_history` and are used to measure persuasion over time.

**Belief drift:** The cosine distance between an agent's initial-belief embedding and its current-belief embedding is recorded each round, giving a numeric measure of how much the agent was persuaded.

---

## Output Files

| File | Description |
|------|-------------|
| `trial_N.json` | Full machine-readable record of trial N (all agent traces, memories, beliefs) |
| `trial_N_summary.txt` | Human-readable round-by-round summary for trial N |
| `overall_summary.txt` | Final beliefs and stance histories across all trials |
| `consensus_summary.txt` | LLM-generated paragraph synthesizing the group's conclusions |
| `stance_evolution.png` | Line chart of each agent's stance label per round |
| `belief_drift.png` | Drift scores showing belief change from round 1 |
| `convergence_curve.png` | Average pairwise stance divergence over rounds |
| `influence_network.png` | Directed graph of who moved whom |
| `conversation_heatmap.png` | Keyword frequency heatmap across agents |
| `venn_diagram.png` | Overlapping themes across trials |

---

## Configuration

To change the debate topic or number of trials/rounds, edit the bottom of `main.py`:

```python
if __name__ == "__main__":
    run_experiments(num_trials=5, rounds=5)
```

To swap in different agent personas, edit `build_agents()` in `main.py`.

---

## Requirements

- Python 3.10+
- OpenAI API key (GPT-4 recommended)
- `openai`, `numpy`, `matplotlib`, `networkx`, `matplotlib-venn`, `python-pptx`

---

## License

MIT
