import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib_venn import venn2, venn3

# This file analyzes the results from completed simulation trials.
# It loads the trial JSON files, extracts key themes, and generates five visualizations:
#   - stance_evolution.png    : how each agent's stance shifted round by round
#   - venn_diagram.png        : which themes appeared across which trials
#   - influence_network.png   : who influenced whom most (directed graph)
#   - belief_drift.png        : how far each agent's belief moved from their starting point
#   - convergence_curve.png   : did the group converge or stay divided over rounds?


STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "that", "this", "we", "i", "you", "they",
    "be", "are", "was", "were", "have", "has", "had", "not", "about",
    "their", "our", "can", "just", "so", "if", "when", "how", "what",
    "more", "really", "think", "feel", "truly", "also", "still", "even",
    "would", "could", "should", "might", "need", "use", "used", "using",
    "do", "did", "does", "been", "being", "all", "some", "any", "no",
    "up", "out", "than", "then", "my", "me", "him", "her", "its", "am",
    "very", "well", "here", "there", "from", "into", "by", "as", "like",
    "will", "get", "much", "too", "point", "way", "good", "right", "see",
    "things", "something", "sure", "bit", "make", "going", "want", "know",
    "actually", "perhaps", "maybe", "quite", "however", "though", "while",
    "across", "them", "us", "those", "these", "each", "both", "only",
    "already", "yes", "no", "own", "agree", "think", "believe", "feel"
}

KEY_THEMES = [
    "critical thinking", "over-rely", "dependency", "misuse", "fairness",
    "creativity", "personalize", "personalized", "guidelines", "boundaries",
    "responsibility", "responsible", "access", "equity", "engagement",
    "brainstorm", "research", "feedback", "learning", "harm",
    "benefits", "risks", "moderation", "integrity", "shortcuts",
    "skills", "independent", "reliance", "support", "adoption"
]


def extract_themes(text: str) -> set:
    # Scans a piece of text and returns which key themes appear in it
    text = text.lower()
    found = set()
    for theme in KEY_THEMES:
        if theme in text:
            found.add(theme)
    return found


def load_trial(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_trial_themes(trial: dict) -> set:
    # Collects all themes mentioned across all agents and rounds in one trial
    themes = set()
    for agent in trial["agents"]:
        for trace in agent["traces"]:
            for field in ["observation_summary", "thought", "influence_analysis",
                          "updated_belief", "updated_goal", "message"]:
                if field in trace:
                    themes |= extract_themes(trace[field])
        themes |= extract_themes(agent.get("final_belief", ""))
        for mem in agent.get("memories", []):
            themes |= extract_themes(mem)
    return themes


def get_trial_stances(trial: dict) -> dict:
    # Returns each agent's stance history (list of supportive/skeptical/balanced per round) for a trial
    return {
        agent["name"]: agent["stance_history"]
        for agent in trial["agents"]
    }


def print_venn_sets(sets: list[set], labels: list[str]):
    all_themes = set().union(*sets)
    shared_all = sets[0].copy()
    for s in sets[1:]:
        shared_all &= s

    unique = [s - set().union(*[sets[j] for j in range(len(sets)) if j != i])
              for i, s in enumerate(sets)]

    print("\n=== Theme Analysis Across Trials ===")
    print(f"\nShared across ALL {len(sets)} trials:")
    for t in sorted(shared_all):
        print(f"  • {t}")

    for i, label in enumerate(labels):
        print(f"\nUnique to {label}:")
        for t in sorted(unique[i]):
            print(f"  • {t}")

    print(f"\nAll themes observed (any trial): {sorted(all_themes)}")


def plot_venn(sets: list[set], labels: list[str], title: str):
    fig, axes = plt.subplots(1, len(sets) - 1, figsize=(6 * (len(sets) - 1), 5))
    if len(sets) - 1 == 1:
        axes = [axes]

    for i in range(len(sets) - 1):
        ax = axes[i]
        a, b = sets[i], sets[i + 1]
        la, lb = labels[i], labels[i + 1]
        venn2([a, b], set_labels=(la, lb), ax=ax)
        ax.set_title(f"{la} vs {lb}")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("venn_diagram.png", dpi=150)
    print("\nSaved: venn_diagram.png")
    plt.close()


def plot_stance_evolution(all_stances: list[dict], trial_labels: list[str]):
    # Generates a chart showing how each agent's stance shifted round by round across all trials
    # Saves the result as stance_evolution.png
    agents = list(all_stances[0].keys())
    stance_to_num = {"supportive": 1, "balanced": 0, "skeptical": -1}
    colors = {"Alice": "#4C9BE8", "Bob": "#E85C5C", "Carol": "#6DBF6D", "David": "#F5A623"}

    fig, axes = plt.subplots(1, len(agents), figsize=(4 * len(agents), 4), sharey=True)
    fig.suptitle("Stance Evolution per Agent Across Trials", fontsize=13, fontweight="bold")

    for ax, agent in zip(axes, agents):
        for i, (stances, label) in enumerate(zip(all_stances, trial_labels)):
            history = stances.get(agent, [])
            nums = [stance_to_num.get(s, 0) for s in history]
            rounds = list(range(1, len(nums) + 1))
            ax.plot(rounds, nums, marker="o", label=label, alpha=0.7,
                    color=plt.cm.tab10(i / len(trial_labels)))

        ax.set_title(agent, color=colors.get(agent, "black"), fontweight="bold")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["skeptical", "balanced", "supportive"])
        ax.set_xlabel("Round")
        ax.set_xticks(range(1, 4))
        ax.grid(True, alpha=0.3)

    axes[-1].legend(title="Trial", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("stance_evolution.png", dpi=150)
    print("Saved: stance_evolution.png")
    plt.close()


AGENT_NAMES = ["Alice", "Bob", "Carol", "David"]
AGENT_COLORS = {"Alice": "#4C9BE8", "Bob": "#E85C5C", "Carol": "#6DBF6D", "David": "#F5A623"}
STANCE_TO_NUM = {"supportive": 1, "balanced": 0, "skeptical": -1}


def parse_influencer(text: str) -> str | None:
    # Looks for an agent name in the influence_analysis text and returns who was mentioned first
    lower = text.lower()
    for name in AGENT_NAMES:
        if name.lower() in lower:
            return name
    return None


def plot_influence_network(trials: list[dict]):
    # Builds a directed graph showing who influenced whom across all trials and rounds.
    # Arrow thickness = how many times that influence was recorded. Arrow goes influencer -> influenced.
    counts: dict[tuple[str, str], int] = {}

    for trial in trials:
        for agent in trial["agents"]:
            influenced = agent["name"]
            for trace in agent["traces"]:
                influencer = parse_influencer(trace.get("influence_analysis", ""))
                if influencer and influencer != influenced:
                    key = (influencer, influenced)
                    counts[key] = counts.get(key, 0) + 1

    if not counts:
        print("No influence data found — skipping influence_network.png")
        return

    G = nx.DiGraph()
    G.add_nodes_from(AGENT_NAMES)
    for (src, dst), weight in counts.items():
        G.add_edge(src, dst, weight=weight)

    _, ax = plt.subplots(figsize=(7, 6))
    pos = nx.circular_layout(G)
    node_colors = [AGENT_COLORS.get(n, "#aaaaaa") for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        G, pos,
        width=[2 + 4 * (w / max_w) for w in edge_weights],
        edge_color="#555555",
        arrows=True,
        arrowsize=25,
        connectionstyle="arc3,rad=0.15",
        ax=ax,
    )
    # Add edge weight labels
    edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

    ax.set_title("Influence Network (arrow = influenced by, thickness = frequency)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("influence_network.png", dpi=150)
    print("Saved: influence_network.png")
    plt.close()


def plot_belief_drift(trials: list[dict]):
    # Shows how far each agent's belief drifted from their starting point, round by round.
    # Drift score: 0.0 = no change, 1.0 = completely different belief.
    # Averaged across all trials so we can see general patterns.
    agents = AGENT_NAMES
    fig, axes = plt.subplots(1, len(agents), figsize=(4 * len(agents), 4), sharey=True)
    fig.suptitle("Belief Drift Per Agent (0 = no change, 1 = fully shifted)", fontsize=13, fontweight="bold")

    for ax, agent_name in zip(axes, agents):
        # Collect drift scores per round across all trials
        round_drifts: dict[int, list[float]] = {}
        for trial in trials:
            for agent in trial["agents"]:
                if agent["name"] != agent_name:
                    continue
                for trace in agent["traces"]:
                    r = trace["round"]
                    d = trace.get("belief_drift", None)
                    if d is not None:
                        round_drifts.setdefault(r, []).append(d)

        if not round_drifts:
            ax.set_title(agent_name, color=AGENT_COLORS.get(agent_name), fontweight="bold")
            ax.text(0.5, 0.5, "no drift data", ha="center", va="center", transform=ax.transAxes)
            continue

        rounds_sorted = sorted(round_drifts.keys())
        means = [np.mean(round_drifts[r]) for r in rounds_sorted]
        stds  = [np.std(round_drifts[r])  for r in rounds_sorted]

        ax.plot(rounds_sorted, means, marker="o", color=AGENT_COLORS.get(agent_name, "#888"), linewidth=2)
        ax.fill_between(rounds_sorted,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=AGENT_COLORS.get(agent_name, "#888"))
        ax.set_title(agent_name, color=AGENT_COLORS.get(agent_name), fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Drift Score")
    plt.tight_layout()
    plt.savefig("belief_drift.png", dpi=150)
    print("Saved: belief_drift.png")
    plt.close()


def plot_convergence_curve(all_stances: list[dict], trial_labels: list[str]):
    # Plots the standard deviation of stances across all agents, per round, per trial.
    # Low std = group is converging. High std = agents still divided.
    fig, ax = plt.subplots(figsize=(8, 4))

    for stances, label in zip(all_stances, trial_labels):
        agents = list(stances.keys())
        num_rounds = max(len(stances[a]) for a in agents)
        std_per_round = []

        for r in range(num_rounds):
            values = []
            for a in agents:
                history = stances[a]
                if r < len(history):
                    values.append(STANCE_TO_NUM.get(history[r], 0))
            if values:
                std_per_round.append(np.std(values))

        rounds = list(range(1, len(std_per_round) + 1))
        ax.plot(rounds, std_per_round, marker="o", label=label, linewidth=2)

    ax.set_title("Convergence Curve — Stance Std Dev Across Agents Per Round",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Std Dev of Stances (lower = more agreement)")
    ax.legend(title="Trial")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=150)
    print("Saved: convergence_curve.png")
    plt.close()


def main():
    paths = sorted(glob.glob("trial_*.json"))
    if not paths:
        print("No trial_*.json files found. Run main.py first.")
        return

    trials = [load_trial(p) for p in paths]
    labels = [f"Trial {t['trial_id']}" for t in trials]
    theme_sets = [get_trial_themes(t) for t in trials]
    all_stances = [get_trial_stances(t) for t in trials]

    print_venn_sets(theme_sets, labels)

    # Venn: compare pairs of consecutive trials (up to 3 at a time for readability)
    plot_venn(theme_sets[:3], labels[:3], "Discussion Themes: Trial Comparison")
    plot_stance_evolution(all_stances, labels)
    plot_influence_network(trials)
    plot_belief_drift(trials)
    plot_convergence_curve(all_stances, labels)


if __name__ == "__main__":
    main()
