import sys
import pandas as pd
import numpy as np
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

from amira_graph_reader import AmiraGraphReader


def main():
    input_file = "/data/projects/md1290_ltp/EDF/ZZ_ONGOING/big_heart_analysis/cleaned_spatial_graph_2.am"
    output_file = "/data/projects/md1290_ltp/EDF/ZZ_ONGOING/big_heart_analysis/output.am"
    run_ordering(input_file, output_file)


def run_ordering(input_file, output_file):
    """ Main function for computing Strahler Order and Topological Generation. """
    
    # Step 1: Read the Amira file
    graph_reader = AmiraGraphReader(input_file)

    print(f"Number of Vertices: {graph_reader.num_vertices}")
    print(f"Number of Edges: {graph_reader.num_edges}")
    print(f"Number of Points: {graph_reader.num_points}")

    # Extract data
    vertex_data = next((d["data"] for d in graph_reader.vertex_data if d["attribute_name"] == "VertexCoordinates"), [])
    edge_data = next((d["data"] for d in graph_reader.edge_data if d["attribute_name"] == "EdgeConnectivity"), [])
    
    if not edge_data:
        print("Error: No edge data found.")
        return
    
    # Step 2: Convert to NetworkX Graph
    edges = [(int(e[0]), int(e[1])) for e in edge_data]
    G = nx.DiGraph(edges)

    # Step 3: Identify Root Nodes (Before Correction)
    root_nodes = [node for node in G.nodes if G.out_degree(node) == 0]
    
    if len(root_nodes) == 0:
        print("Error: No root node found. The graph may not be a tree.")
        return
    elif len(root_nodes) > 1:
        print(f"Warning: Multiple root nodes detected ({len(root_nodes)}).")
        print(f"Identified root nodes: {root_nodes}")

        # Ask user to select the correct root node
        root_node = user_select_root(root_nodes)
    else:
        root_node = root_nodes[0]

    print(f"Initial Root Node: {root_node}")

    # Step 4: Validate & Correct Edges
    corrected_edges, bad_edge_indices = find_bad_edges(edges, root_node, update_root=True)
    print(f"Number of bad edges found: {len(bad_edge_indices)}")

    # Step 5: Recompute Root Nodes **After** Edge Correction
    G_corrected = nx.DiGraph(corrected_edges)

    updated_leaf_nodes = [node for node in G_corrected.nodes if G_corrected.in_degree(node) == 0]
    updated_root_nodes = [node for node in G_corrected.nodes if G_corrected.out_degree(node) == 0]

    print(f"Updated Root Nodes: {updated_root_nodes}")
    print(f"Updated Leaf Nodes: {updated_leaf_nodes}")

    # Step 6: Visualize the Corrected Graph
    # visualize_graph(vertex_data, corrected_edges, updated_root_nodes, updated_leaf_nodes)

    # Step 7: Compute Strahler Order
    strahler_order, strahler_order_list = calculate_strahler_order(G_corrected, edge_data)


    graph_reader.add_EDGE_data('strahler', 'int', strahler_order_list)
    graph_reader.write_file(output_file)

    import pdb; pdb.set_trace()

    # Step 8: Compute Topological Generation
    generations = topological_generation(G_corrected)

    graph_reader.add_EDGE_data('topo', generations)

    graph_reader.write_file(output_file)
    print(f"Results saved to {output_file}")


def user_select_root(root_nodes):
    """ Prompts user to confirm or update the root node. """
    while True:
        try:
            print("\nAvailable Root Nodes:", root_nodes)
            root_node = int(input("Enter the correct Root Node ID (as written in Amira): "))
            # root_node -= 1 # In python first element start at 0 and not 1
            if not root_node:
                sys.exit("Nothing entered")
            
            return root_node
            
            # if root_node in root_nodes:
            #     return root_node
            # else:
            #     confirm = input("Are you really sure that is the root nodeID, double check it in amira if you say yes now I will re-do the newtork with this root node so if you are wrong it will be sad: [y]").strip().lower()
            #     if not 'n' in confirm:
            #         return root_node
        except ValueError:
            print("Invalid input. Please enter a valid numeric Root Node ID.")


def find_bad_edges(edge_nodes, root_node, update_root=False):
    """
    Identifies and corrects incorrect edges in a directed graph.

    :param edge_nodes: List of (source, target) edges.
    :param root_node: The root node ID.
    :param update_root: Boolean flag to update root connections if needed.
    :return: Corrected edges and indices of bad edges.
    """

    edge_nodes = np.array(edge_nodes)  # Convert to NumPy array for easier manipulation

    # Step 1: Handle update_root condition
    if update_root:
        root_indices = np.where(edge_nodes[:, 0] == root_node)[0]  # Find edges where root is source
        for i in root_indices:
            edge_nodes[i] = [edge_nodes[i][1], edge_nodes[i][0]]  # Swap (flip) the edge direction

    edge_nodes_temp = edge_nodes.copy()

    # Step 2: Find incorrect edges
    bad_edge_indices = []
    strahler = 1

    while len(edge_nodes_temp) > 1:
        unique_nodes, counts = np.unique(edge_nodes_temp, return_counts=True)  # Get node occurrences
        terminal_nodes = unique_nodes[counts == 1]  # Nodes appearing only once (leaf nodes)

        
        # Identify bad edges (where target is in terminal nodes)
        bad_mask = np.isin(edge_nodes_temp[:, 1], terminal_nodes)  # Find edges pointing to terminal nodes
        bad_edges = edge_nodes_temp[bad_mask]  # Extract bad edges
        bad_edge_indices.extend([np.where((edge_nodes == edge).all(axis=1))[0][0] for edge in bad_edges])

        # Identify good edges (where source is in terminal nodes)
        good_mask = np.isin(edge_nodes_temp[:, 0], terminal_nodes)

        # Remove identified edges
        remove_indices = np.where(bad_mask | good_mask)[0]
        edge_nodes_temp = np.delete(edge_nodes_temp, remove_indices, axis=0)

        # Add root edge back to stabilize
        edge_nodes_temp = np.vstack([edge_nodes_temp, edge_nodes[np.where(edge_nodes[:, 1] == root_node)]])

        strahler += 1  # Increase Strahler order step
        
    # Step 3: Correct flipped edges
    bad_edge_indices = list(set(bad_edge_indices) - set(np.where(edge_nodes[:, 1] == root_node)[0]))
    for i in bad_edge_indices:
        edge_nodes[i] = [edge_nodes[i][1], edge_nodes[i][0]]  # Flip incorrect edges

    return edge_nodes.tolist(), bad_edge_indices





def visualize_graph(vertex_data, edge_data, root_nodes, leaf_nodes):
    """Visualizes the spatial graph using Graphviz's 'dot' layout for MATLAB-like hierarchical display with node labels."""
    
    if not vertex_data or not edge_data:
        print("Warning: No data available for visualization.")
        return

    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(len(vertex_data)))  # Add nodes
    G.add_edges_from(edge_data)

    # Define node colors explicitly
    node_colors = []
    for node in G.nodes:
        if node in root_nodes:
            node_colors.append("red")  # Root nodes are red
        elif node in leaf_nodes:
            node_colors.append("green")  # Leaf nodes are green
        else:
            node_colors.append("blue")  # Intermediate nodes are blue
            
    # for idx, node in enumerate(G.nodes):
    #     if node == 81:
    #         node_colors[idx] = "red"
            
            
    # Compute node positions using Graphviz's 'dot' layout
    pos = graphviz_layout(G, prog="dot")  # Uses Graphviz to generate a top-down tree

    # Generate labels for each node (Node ID)
    labels = {node: str(node) for node in G.nodes()}

    # Plot the graph
    plt.figure(figsize=(12, 10))  # Slightly larger figure
    nx.draw(
        G, pos, labels=labels, with_labels=True, node_size=50, node_color=node_colors,
        font_size=5, edge_color="lightblue", alpha=0.6, width=0.7
    )

    plt.title("Spatial Graph with Node IDs")
    plt.show(block=False)
    
    
    
    
def calculate_strahler_order(G, edge_data):
    """Computes Strahler order for each node in an inverted graph (root at bottom, leaves at top).
    
    Ensures that the Strahler order list follows the order of `edge_data`.
    """

    # Initialize all nodes with default order 1
    strahler_order = {node: 1 for node in G.nodes}

    # Process nodes in **topological order** (from roots to leaves)
    nodes_sorted = list(nx.topological_sort(G))  # Process from root to leaves

    for node in nodes_sorted:
        parents = list(G.predecessors(node))  # Get parent nodes

        if parents:
            # Get the Strahler orders of the parents
            parent_orders = [strahler_order.get(parent, 1) for parent in parents]

            max_order = max(parent_orders)
            # If multiple parents have the max order, increment it
            if parent_orders.count(max_order) > 1:
                strahler_order[node] = max_order + 1
            else:
                strahler_order[node] = max_order

    # **Align the Strahler order list with the edge connectivity order**
    strahler_order_list = []
    for edge in edge_data:
        source, target = edge[:2]  # Extract source and target nodes
        strahler_order_list.append(strahler_order[source])  # Store Strahler order of the target node

    return strahler_order, strahler_order_list  # Return dict & ordered list




def topological_generation(G):
    """ Computes topological generation numbers for nodes. """
    levels = {}
    for i, node in enumerate(nx.topological_sort(G)):
        levels[node] = i
    return levels



if __name__ == "__main__":
    main()
