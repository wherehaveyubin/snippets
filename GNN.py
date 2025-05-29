import networkx as nx
import matplotlib.pyplot as plt

## Basic graph representation

## Undirected graph
# Create a graph
G = nx.Graph()

# Add nodes (representing people)
G.add_node("Yubin")
G.add_node("Youngho")
G.add_node("Yumin")
G.add_node("Woojeong")
G.add_node("Rey")
G.add_node("Coin")

# Add edges (representing friendship)
G.add_edges_from([("Yubin", "Youngho"), ("Yubin", "Yumin"), ("Yubin", "Woojeong"), ("Yubin", "Rey"), ("Youngho", "Coin"), ("Yumin", "Woojeong"), ("Yumin", "Rey")])

# Make a plot
plt.axis('off')
nx.draw_networkx(G,
                 pos=nx.spring_layout(G, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=6,
                 font_color='white'
                 )


## Directed graph
# Create a graph
DG = nx.DiGraph() 
DG.add_edges_from([("Yubin", "Youngho"), ("Yubin", "Yumin"), ("Yubin", "Woojeong"), ("Yubin", "Rey"), ("Youngho", "Coin"), ("Yumin", "Woojeong"), ("Yumin", "Rey")])

# Make a plot
plt.axis('off')
nx.draw_networkx(DG,
                 pos=nx.spring_layout(G, seed=0),
                 node_size=600,
                 cmap='coolwarm',
                 font_size=6,
                 font_color='white'
                 )


## GNN example

# 1. Plot the overall graph structure with node features
def plot_graph_structure_detailed():
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes with feature information
    G.add_nodes_from([
        ("A", {"feature": "Feature_A"}),
        ("B", {"feature": "Feature_B"}),
        ("C", {"feature": "Feature_C"}),
        ("D", {"feature": "Feature_D"}),
        ("E", {"feature": "Feature_E"}),
        ("F", {"feature": "Feature_F"}),
        ("G", {"feature": "Feature_G"}),
        ("H", {"feature": "Feature_H"})
    ])

    # Add directed edges representing message flow direction
    G.add_edges_from([
        ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E"),
        ("D", "F"), ("E", "F"), ("E", "G"), ("F", "H"), ("G", "H")
    ])
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for consistent positioning
    nx.draw(G, pos, with_labels=True, node_color='lightpink', node_size=2000, font_size=12, arrows=True)
    
    # Annotate each node with its feature
    labels = {node: f"{node}\n{data['feature']}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    plt.title("1. Graph Structure: Nodes with Features and Directed Edges")
    plt.show()
