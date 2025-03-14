import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Define states
states = ["C1", "C2"]

# Add state transition probabilities
transitions = {
    ("C1", "C1"): 0.9,  # P(C1|C1)
    ("C1", "C2"): 0.1,  # P(C2|C1)
    ("C2", "C2"): 0.8,  # P(C2|C2)
    ("C2", "C1"): 0.2,  # P(C1|C2)
}

# Add nodes and edges
for (start, end), prob in transitions.items():
    G.add_edge(start, end, weight=prob)

# Observation probabilities (connect to virtual "observation" nodes)
observations = {
    "C1": {"H": 0.9, "T": 0.1},  # Observations for state C1
    "C2": {"H": 0.5, "T": 0.5},  # Observations for state C2
}

# Add observation nodes
for state, obs_probs in observations.items():
    for obs, prob in obs_probs.items():
        obs_label = f"{obs}_{state}"  # Ensure unique observation labels
        G.add_edge(state, obs_label, weight=prob)

# Set positions for states and observations
pos = {
    "C1": (0, 1), 
    "C2": (2, 1),  # State positions
    "H_C1": (-1, 0), "T_C1": (1, 0),  # Observation positions for state C1
    "H_C2": (3, 0), "T_C2": (1, 0)   # Observation positions for state C2
}

# Draw the graph
plt.figure(figsize=(10, 7))
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="black")
nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")

# Save the graph as an image file
output_file = "state_machine.png"
plt.title("Hidden Markov Model State Machine")
plt.axis("off")
plt.savefig(output_file, format="png", dpi=300)
print(f"State machine diagram saved as {output_file}.")
