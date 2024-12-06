import networkx as nx
import random
from cal_distance import path_calculator
import os
import math

def get_random_node_pairs(graph, num_pairs):
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("The graph must contain at least two nodes")
    node_pairs = set()
    while len(node_pairs) < num_pairs:
        pair = tuple(random.sample(nodes, 2))
        if pair not in node_pairs:
            node_pairs.add(pair)
    return list(node_pairs)

if __name__ == "__main__":
    # Create directories if they don't exist
    for dir_path in [
        "basic_selection/pre_compute distance",
        "constrained_selection/pre_compute distance",
        "partitioning_selection/pre_compute distance"
    ]:
        os.makedirs(dir_path, exist_ok=True)

    networks = ["reddit_body", "reddit_title"]
    selections = {
        'basic_selection': ['basic_centrality_selection', 'basic_degree_selection'],
        'constrained_selection': ['constrained_centrality_selection', 'constrained_degree_selection'],
        'partitioning_selection': ['partitioning_centrality_selection', 'partitioning_degree_selection']
    }

    cal_path = path_calculator()

    for network in networks:
        graph_file_path = "data to graph/" + network + ".gexf"
        print(f"\nProcessing network: {network}")
        
        try:
            # Load graph
            graph = nx.read_gexf(graph_file_path)
            print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            
            # Generate random pairs
            random_pairs = get_random_node_pairs(graph, 100)
            print(f"\n{'Selection Method':<40} {'Error Value':<10}")
            print("-" * 50)
            
            for selection_type, methods in selections.items():
                for method in methods:
                    try:
                        # Fix the file path format to match actual files
                        dis_file_path = f"{selection_type}/pre_compute distance/{method}_{network}.json"
                        if cal_path.load_distances(graph, dis_file_path):
                            print(f"\nComputing error for {method}")
                            error = cal_path.compute_error(random_pairs)
                            if not math.isnan(error):
                                print(f"{method:<40} {error:.4f}")
                            else:
                                print(f"{method:<40} No valid pairs found")
                    except Exception as e:
                        print(f"{method:<40} Error: {str(e)}")
                        
        except Exception as e:
            print(f"Error processing {network}: {str(e)}")