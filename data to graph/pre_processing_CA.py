import networkx as nx

def create_road_network(file_path, output_path):
    """
    Convert California road network data into a NetworkX graph and save it.
    The file contains header comments (lines starting with #) and 
    space-separated FromNodeId ToNodeId pairs.
    
    Parameters:
    file_path (str): Path to the roadNet-CA.txt file
    output_path (str): Path where to save the graph
    
    Returns:
    nx.DiGraph: NetworkX directed graph representing the road network
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Read the file and create the graph
    with open(file_path, 'r') as f:
        # Skip comment lines starting with #
        for line in f:
            if line.startswith('#'):
                continue
            
            # Split the line into source and target nodes
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)
    
    # Save the graph
    nx.write_gexf(G, output_path)
    
    return G

# Example usage
if __name__ == "__main__":
    input_path = "roadNet-CA.txt"
    graph_path = "CA.gexf"
    
    # Create and save the graph
    G = create_road_network(input_path, graph_path)
    
    # Print basic network statistics
    print("\nCalifornia Road Network Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Average in-degree: {sum(dict(G.in_degree()).values()) / G.number_of_nodes():.2f}")
    
    # Additional road network specific metrics
    print("\nNetwork properties:")
    print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
    print(f"Number of strongly connected components: {nx.number_strongly_connected_components(G)}")