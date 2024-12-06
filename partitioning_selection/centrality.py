import networkx as nx
from typing import List
from pathlib import Path
from tqdm import tqdm
from networkx.algorithms.community import kernighan_lin_bisection
import random


class partitioning_centrality_selection:
    """
    A class to select landmarks in networks using closeness centrality and graph partitioning.
    First partitions the graph, then selects nodes with lowest closeness centrality from each partition.
    """

    def __init__(self, gexf_path: str = None):
        """
        Initialize the LandmarkSelector.

        Args:
            gexf_path: Path to the .gexf network file. If provided, loads the network immediately.
        """
        self.G = None
        self.landmarks = []
        if gexf_path:
            self.load_network(gexf_path)

    def load_network(self, network_path: str) -> None:
        """
        Load network from a .gexf file.

        Args:
            network_path: Path to the .gexf file
        """
        path = Path(network_path)
        if path.suffix != '.gexf':
            raise ValueError("Network file must be in .gexf format")

        if not path.exists():
            raise FileNotFoundError(f"Network file not found: {network_path}")

        try:
            self.G = nx.read_gexf(network_path)
            print(f"Network loaded successfully: {len(self.G.nodes())} nodes, "
                  f"{len(self.G.edges())} edges")
        except Exception as e:
            raise Exception(f"Error loading network: {str(e)}")

    def select_landmarks(self, num_landmarks: int) -> List[int]:
        """
        Select landmarks using partitioning and centrality.
        1. Partition the graph using Louvain method
        2. For each partition, select the node with lowest approximate closeness centrality
        
        Args:
            num_landmarks: Number of landmarks to select

        Returns:
            List of selected landmark node IDs
        """
        if not self.G:
            raise ValueError("No network loaded. Call load_network() first.")

        print("Partitioning graph...")
        G_undirected = self.G.to_undirected()
        
        # Use recursive bisection to create num_landmarks partitions
        partitions = {node: 0 for node in self.G.nodes()}  # Start with all nodes in one partition
        partition_to_nodes = {0: list(self.G.nodes())}
        
        current_partitions = 1
        # Calculate total number of bisections needed
        target_partitions = num_landmarks
        num_bisections = max(0, target_partitions - 1)  # -1 because we start with one partition
        
        with tqdm(total=num_bisections, desc="Creating partitions") as pbar:
            while current_partitions < num_landmarks:
                # Find largest partition
                largest_partition = max(partition_to_nodes.keys(), 
                                    key=lambda k: len(partition_to_nodes[k]))
                
                if len(partition_to_nodes[largest_partition]) <= 1:
                    break
                    
                # Create subgraph of largest partition
                subgraph = G_undirected.subgraph(partition_to_nodes[largest_partition])
                
                # Bisect the partition
                try:
                    sets = kernighan_lin_bisection(subgraph)
                    if not sets:
                        continue
                    set1, set2 = sets
                    
                    # Remove old partition and add two new ones
                    nodes = partition_to_nodes[largest_partition]
                    del partition_to_nodes[largest_partition]
                    
                    new_partition1 = current_partitions
                    new_partition2 = current_partitions + 1
                    
                    partition_to_nodes[new_partition1] = list(set1)
                    partition_to_nodes[new_partition2] = list(set2)
                    
                    for node in set1:
                        partitions[node] = new_partition1
                    for node in set2:
                        partitions[node] = new_partition2
                        
                    current_partitions += 2
                    pbar.update(1)
                except:
                    continue
                
        print(f"Created {len(partition_to_nodes)} partitions")
        
        num_partitions = len(partition_to_nodes)
        landmarks_per_partition = max(1, num_landmarks // num_partitions)
        
        print("Computing approximate closeness centrality...")
        # Sample nodes for approximation
        sample_size = min(100, len(self.G))
        sample_nodes = random.sample(list(self.G.nodes()), sample_size)
        
        centrality_scores = {}
        with tqdm(total=len(sample_nodes), desc="Calculating centrality") as pbar:
            for source in sample_nodes:
                distances = nx.single_source_shortest_path_length(self.G, source)
                for node in distances:
                    if node not in centrality_scores:
                        centrality_scores[node] = 0
                    centrality_scores[node] += distances[node]
                pbar.update(1)

        # Normalize scores
        for node in centrality_scores:
            centrality_scores[node] /= len(sample_nodes)

        print(f"Selecting {landmarks_per_partition} landmarks from each partition...")
        self.landmarks = []
        with tqdm(total=num_landmarks, desc="Selecting landmarks") as pbar:
            for partition_id in sorted(partition_to_nodes.keys()):
                partition_nodes = partition_to_nodes[partition_id]
                # Sort nodes in partition by centrality
                partition_nodes.sort(key=lambda x: centrality_scores.get(x, float('inf')))
                
                # Select top nodes from partition
                selected = partition_nodes[:landmarks_per_partition]
                self.landmarks.extend(selected)
                pbar.update(len(selected))
                
                if len(self.landmarks) >= num_landmarks:
                    break

        # Trim to exact number if needed
        self.landmarks = self.landmarks[:num_landmarks]
        return self.landmarks