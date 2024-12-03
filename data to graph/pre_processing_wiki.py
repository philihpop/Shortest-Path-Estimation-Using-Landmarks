import networkx as nx

def create_wiki_network(file_path, output_path):
    """
    Convert Wikipedia Talk network data into a NetworkX graph and save it.
    The file contains header comments (lines starting with #) and 
    space-separated FromNodeId ToNodeId pairs.
    
    Parameters:
    file_path (str): Path to the WikiTalk.txt file
    output_path (str): Path where to save the graph
    
    Returns:
    nx.DiGraph: NetworkX directed graph representing the network
    """
    # Create a directed graph since edges represent directed edits
    G = nx.DiGraph()
    
    # Read the file and create the graph
    with open(file_path, 'r') as f:
        # Skip comment lines starting with #
        for line in f:
            if line.startswith('#'):
                continue
            
            # Split the line into source and target nodes
            # Using split() with no arguments handles multiple spaces
            source, target = map(int, line.strip().split())
            G.add_edge(source, target)
    
    # Save the graph
    nx.write_gexf(G, output_path)
    
    return G

# Example usage
if __name__ == "__main__":
    input_path = "WikiTalk.txt"
    graph_path = "wiki.gexf"
    
    # Create and save the graph
    G = create_wiki_network(input_path, graph_path)
    
    # Print basic network statistics
    print("\nWiki-Talk Network Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average out-degree: {sum(dict(G.out_degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Average in-degree: {sum(dict(G.in_degree()).values()) / G.number_of_nodes():.2f}")
    
    # Print first few edges to verify correct reading
    print("\nFirst 5 edges in the graph:")
    for i, edge in enumerate(list(G.edges())[:5]):
        print(f"Edge {i+1}: {edge}")
    
    # Additional network metrics
    print("\nNetwork properties:")
    print(f"Is strongly connected: {nx.is_strongly_connected(G)}")
    print(f"Number of strongly connected components: {nx.number_strongly_connected_components(G)}")
    print(f"Is weakly connected: {nx.is_weakly_connected(G)}")
    print(f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}")