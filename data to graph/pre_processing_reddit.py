import networkx as nx
import pandas as pd

def create_network_from_tsv(file_path, output_path):
    """
    Convert TSV file containing Reddit cross-post data into a NetworkX graph
    and save it to a file.
    
    Parameters:
    file_path (str): Path to the TSV file
    output_path (str): Path where to save the graph
    
    Returns:
    nx.Graph: NetworkX graph representing the network
    """
    # Read TSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Create an empty directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges from the data
    for _, row in df.iterrows():
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        
        # Add edge (nodes will be automatically added if they don't exist)
        G.add_edge(source, target)
    
    # Save the graph
    nx.write_gexf(G, output_path)
    
    return G

def load_network(file_path):
    """
    Load a previously saved network from a file.
    
    Parameters:
    file_path (str): Path to the saved graph file
    
    Returns:
    nx.Graph: Loaded NetworkX graph
    """
    return nx.read_gexf(file_path)

# Example usage
if __name__ == "__main__":
    tsv_path = "soc-redditHyperlinks-title.tsv"
    graph_path = "reddit_title.gexf"
    
    # Create and save the graph
    G = create_network_from_tsv(tsv_path, graph_path)
    
    # Print basic information about the graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Example of loading the saved graph
    loaded_G = load_network(graph_path)
    print("\nLoaded graph:")
    print(f"Number of nodes: {loaded_G.number_of_nodes()}")
    print(f"Number of edges: {loaded_G.number_of_edges()}")

    tsv_path = "soc-redditHyperlinks-body.tsv"
    graph_path = "reddit_body.gexf"
    
    # Create and save the graph
    G = create_network_from_tsv(tsv_path, graph_path)
    
    # Print basic information about the graph
    print(f"\nNumber of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Example of loading the saved graph
    loaded_G = load_network(graph_path)
    print("\nLoaded graph:")
    print(f"Number of nodes: {loaded_G.number_of_nodes()}")
    print(f"Number of edges: {loaded_G.number_of_edges()}")