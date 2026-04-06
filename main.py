import json
from agent import Agent

# This file runs the full multi-agent simulation.
# It builds the 4 agents, runs them through multiple rounds of discussion,
# and saves the results to JSON and text summary files.


def build_agents():
    # Creates the 4 agents with their starting personas, beliefs, and goals
    return [
        Agent(
            "Alice",
            "Alice is generally optimistic about educational technology, though she is genuinely open to concerns about equity and over-reliance.",
            "AI tools can support learning and creativity, but I'm still figuring out where the right limits are.",
            "Promote the benefits of AI in education while staying open to criticism that might sharpen my view."
        ),
        Agent(
            "Bob",
            "Bob is skeptical about AI in schools, but he acknowledges that some thoughtful applications could be genuinely useful.",
            "AI tools carry real risks of dependency and misuse, though I can imagine responsible use cases.",
            "Surface the risks and blind spots in overly optimistic views, while remaining open to evidence that changes my mind."
        ),
        Agent(
            "Carol",
            "Carol is balanced and tries to mediate.",
            "AI tools are useful, but schools need clear boundaries.",
            "Mediate between both sides and push the group toward balanced guidelines."
        ),
        Agent(
            "David",
            "David is mildly supportive but cautious.",
            "AI tools can help students, but only when used responsibly.",
            "Support useful adoption of AI while emphasizing moderation and responsibility."
        )
    ]


def save_trial_json(trial_id: int, topic: str, agents: list[Agent]) -> None:
    # Saves the full simulation data for one trial as a JSON file (used by analyze.py later)
    trial_data = {
        "trial_id": trial_id,
        "topic": topic,
        "agents": []
    }

    for agent in agents:
        trial_data["agents"].append({
            "name": agent.name,
            "persona": agent.persona,
            "initial_belief": agent.initial_belief,
            "final_belief": agent.current_belief,
            "final_goal": agent.current_goal,
            "stance_history": agent.stance_history,
            "traces": agent.round_traces,
            "memories": [str(m) for m in agent.memory_stream.get_all()]
        })

    with open(f"trial_{trial_id}.json", "w", encoding="utf-8") as f:
        json.dump(trial_data, f, indent=2, ensure_ascii=False)


def save_trial_summary(trial_id: int, topic: str, agents: list[Agent]) -> None:
    # Saves a human-readable text summary of the trial (beliefs, stances, round-by-round traces)
    with open(f"trial_{trial_id}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"TRIAL {trial_id}\n")
        f.write(f"Topic: {topic}\n\n")

        for agent in agents:
            f.write(f"Agent: {agent.name}\n")
            f.write(f"Persona: {agent.persona}\n")
            f.write(f"Initial belief: {agent.initial_belief}\n")
            f.write(f"Final belief: {agent.current_belief}\n")
            f.write(f"Final goal: {agent.current_goal}\n")
            f.write(f"Stance history: {' -> '.join(agent.stance_history)}\n")
            f.write("Round traces:\n")

            for trace in agent.round_traces:
                f.write(f"  Round {trace['round']}\n")
                f.write(f"    Retrieved memories: {trace['selected_memories']}\n")
                f.write(f"    Observation: {trace['observation_summary']}\n")
                f.write(f"    Thought: {trace['thought']}\n")
                f.write(f"    Influence: {trace['influence_analysis']}\n")
                f.write(f"    Updated belief: {trace['updated_belief']}\n")
                f.write(f"    Updated goal: {trace['updated_goal']}\n")
                f.write(f"    Stance: {trace['stance']}\n")
                f.write(f"    Message: {trace['message']}\n")
            f.write("\n")


def run_one_simulation(trial_id: int, rounds: int = 5):
    # Runs a single trial: agents take turns speaking for N rounds, then reflections are saved
    topic = "whether students should use AI tools in education"
    agents = build_agents()

    print("\n==============================")
    print(f"TRIAL {trial_id}")
    print(f"Topic: {topic}")
    print("==============================\n")

    previous_round_messages = {}

    for round_id in range(1, rounds + 1):
        print(f"===== Round {round_id} =====")
        current_round_messages = {}

        for speaker in agents:
            message = speaker.speak(topic, current_round=round_id, previous_round_messages=previous_round_messages)
            current_round_messages[speaker.name] = message
            trace = speaker.round_traces[-1]

            print(f"\n{speaker.name}")
            print(f"Belief: {trace['updated_belief']}")
            print(f"Goal: {trace['updated_goal']}")
            print(f"Observation: {trace['observation_summary']}")
            print(f"Thought: {trace['thought']}")
            print(f"Influence: {trace['influence_analysis']}")
            print(f"Message: {message}")

            for listener in agents:
                if listener.name != speaker.name:
                    listener.observe(speaker.name, message, round_id)

        print("\n--- Reflections after this round ---")
        for agent in agents:
            reflection = agent.reflect(current_round=round_id)
            print(f"{agent.name}: {reflection}")

        previous_round_messages = current_round_messages
        print()

    print("===== Final Stance Evolution =====")
    for agent in agents:
        history = " → ".join(agent.stance_history) if agent.stance_history else "No stance recorded"
        print(f"{agent.name}: {history}")

    print("\n===== Final Beliefs =====")
    for agent in agents:
        print(f"{agent.name}: {agent.current_belief}")

    print("\n===== Final Goals =====")
    for agent in agents:
        print(f"{agent.name}: {agent.current_goal}")

    save_trial_json(trial_id, topic, agents)
    save_trial_summary(trial_id, topic, agents)

    return agents


def save_overall_summary(all_agents_by_trial: list[list[Agent]], num_trials: int, rounds: int) -> None:
    # Writes a combined summary across all trials to overall_summary.txt
    with open("overall_summary.txt", "w", encoding="utf-8") as f:
        f.write("Overall Experiment Summary\n")
        f.write("==========================\n\n")
        f.write(f"Total trials: {num_trials}\n")
        f.write(f"Rounds per trial: {rounds}\n\n")

        for i, agents in enumerate(all_agents_by_trial, start=1):
            f.write(f"Trial {i}\n")
            for agent in agents:
                f.write(
                    f"  {agent.name}\n"
                    f"    Final belief: {agent.current_belief}\n"
                    f"    Final goal: {agent.current_goal}\n"
                    f"    Stance history: {' -> '.join(agent.stance_history)}\n"
                )
            f.write("\n")


def run_experiments(num_trials: int = 5, rounds: int = 5):
    # Entry point: runs N trials and saves an overall summary when done
    all_agents_by_trial = []

    for trial_id in range(1, num_trials + 1):
        agents = run_one_simulation(trial_id, rounds=rounds)
        all_agents_by_trial.append(agents)

    save_overall_summary(all_agents_by_trial, num_trials, rounds)


if __name__ == "__main__":
    run_experiments(num_trials=5, rounds=5)