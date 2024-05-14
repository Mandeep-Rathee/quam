
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors



# graph_data = {
#     "1": {"nbh": [2, 3, 4], "edge_weights": [0.1, 0.4, 0.5]},
#     "2": {"nbh": [3, 6, 1], "edge_weights": [0.5, 0.1, 0.9]}
# }
# node_weights = {"1": 0.9, "2": 0.7, "3": 1.5}

def plot_graph(graph_data, node_weights, path, iteration):
# Create a graph
    G = nx.Graph()

    # Add nodes and edges with weights
    for source, data in graph_data.items():
        source_node = int(source)
        neighbors = data["nbh"]
        edge_weights = data["aff_score"]

        # Add source node
        G.add_node(source_node)

        # Add node weight if available, otherwise default to 0
        if source in node_weights:
            G.nodes[source_node]["weight"] = node_weights.get(source, None)

        # Add edges and their weights
        for neighbor, weight in zip(neighbors, edge_weights):
            neighbor_node = int(neighbor)
            if iteration!=0 and neighbor in node_weights:
                G.add_node(neighbor_node)
                G.nodes[neighbor_node]["weight"] = node_weights.get(neighbor,None)
            G.add_edge(source_node, neighbor_node, weight=weight)

    # Plot the graph
    plt.figure(figsize=(10, 8))  # Increase the size of the plot
    pos = nx.spring_layout(G,k=0.5,seed=42)  # Positioning the nodes using the spring layout



    nodes_with_weight = [node for node in G.nodes() if "weight" in G.nodes[node] and G.nodes[node]["weight"] is not None]
    nodes_without_weight = [node for node in G.nodes() if "weight" not in G.nodes[node] or G.nodes[node]["weight"] is None]


    if nodes_with_weight:
        node_colors = [G.nodes[node]["weight"] for node in nodes_with_weight]
        norm = mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
        node_scalar_map = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=norm)
        nx.draw(G, pos, nodelist=nodes_with_weight, with_labels=False, node_size=1000, node_color=node_colors, edgecolors='black', cmap=plt.cm.Reds, font_size=6,font_family='serif')
        nx.draw_networkx_labels(G, pos, labels={node: f'{G.nodes[node]["weight"]:.3f}' for node in nodes_with_weight}, font_color='black', font_size=8, verticalalignment='center', horizontalalignment='center')

    nx.draw(G, pos, nodelist=nodes_without_weight, with_labels=False, node_size=1000, node_color='none', edgecolors='black', font_size=6, font_family='serif')


    # Edge color based on weight
    edge_colors = [d["weight"] for _, _, d in G.edges(data=True)]
    norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_scalar_map = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2)


    # Color bar for node weights
    node_cbar = plt.colorbar(node_scalar_map, label='Rel score', location='left', ax=plt.gca())
    node_cbar.ax.yaxis.set_label_position('left')  # Set position to left side

    # Color bar for edge weights
    edge_cbar = plt.colorbar(edge_scalar_map, label='aff score',location='right', ax=plt.gca())
    edge_cbar.ax.yaxis.set_label_position('right')  # Set position to right side

    plt.axis('off')
    #plt.show()

    plt.savefig(path, bbox_inches='tight')
