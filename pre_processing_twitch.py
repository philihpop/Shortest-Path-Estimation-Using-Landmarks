import networkx as nx
import pandas as pd

def create_twitch_network(file_path, output_path):
    """
    Convert Twitch edges CSV file into a NetworkX graph and save it.
    The CSV file has headers 'numeric_id_1,numeric_id_2' and values separated by commas.
    
    Parameters:
    file_path (str): Path to the CSV file containing edges
    output_path (str): Path where to save the graph
    
    Returns:
    nx.Graph: NetworkX graph representing the network
    """
    # Read CSV file with header
    # The header is 'numeric_id_1,numeric_id_2'
    df = pd.read_csv(file_path)
    
    # Create an empty undirected graph
    G = nx.Graph()
    
    # Add edges from the data (nodes will be automatically created)
    edge_list = list(zip(df['numeric_id_1'], df['numeric_id_2']))
    G.add_edges_from(edge_list)
    
    # Save the graph
    nx.write_gexf(G, output_path)
    
    return G

# Example usage
if __name__ == "__main__":
    csv_path = "large_twitch_edges.csv"
    graph_path = "twitch.gexf"
    
    # Create and save the graph
    G = create_twitch_network(csv_path, graph_path)
    
    # Print basic network statistics
    print("\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    # Print first few edges to verify correct reading
    print("\nFirst 5 edges in the graph:")
    for i, edge in enumerate(list(G.edges())[:5]):
        print(f"Edge {i+1}: {edge}")