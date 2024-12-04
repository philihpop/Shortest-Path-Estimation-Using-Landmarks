import networkx as nx
from typing import List, Set
from pathlib import Path


class constrained_centrality_selection:
    """
    A class to select landmarks in networks using closeness centrality based constrained heuristic approach.
    Selects nodes with lower closeness centrality as landmarks.
    For each selected landmark l, nodes at distance h or less from l are removed from consideration.
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

    def select_landmarks(self, num_landmarks: int, h: int = 1) -> List[int]:
        """
        Select landmarks iteratively based on closeness centrality.
        Nodes with lower closeness centrality are preferred as landmarks.
        For each selected landmark l, remove all nodes at distance h or less from l.

        Args:
            num_landmarks: Number of landmarks to select
            h: Distance parameter (nodes within distance h from a landmark are removed)

        Returns:
            List of selected landmark node IDs
        """
        if not self.G:
            raise ValueError("No network loaded. Call load_network() first.")

        # Calculate closeness centrality for all nodes
        closeness_centrality = nx.closeness_centrality(self.G)

        # Sort nodes by closeness centrality (ascending order - lower values first)
        ranked_nodes = sorted(closeness_centrality.keys(),
                              key=lambda k: closeness_centrality[k])

        self.landmarks = []
        available_nodes = set(self.G.nodes())

        while len(self.landmarks) < num_landmarks and available_nodes:
            # Select node with lowest closeness centrality from available nodes
            for node in ranked_nodes:
                if node in available_nodes:
                    landmark = node
                    break
            else:
                # No more available nodes
                break

            self.landmarks.append(landmark)

            # Find and remove all nodes within distance h from the landmark
            to_remove = set()
            paths = nx.single_source_shortest_path_length(self.G, landmark)

            for node, distance in paths.items():
                if distance <= h:
                    to_remove.add(node)

            # Remove nodes from consideration
            available_nodes -= to_remove

            # Print information about selected landmark
            print(f"Selected landmark {landmark} with closeness centrality: {closeness_centrality[landmark]:.4f}")
            print(f"Removed {len(to_remove)} nodes within distance {h}")

        return self.landmarks

    def save_landmarks(self, output_path: str) -> None:
        """
        Save selected landmarks to a file.

        Args:
            output_path: Path to save the landmarks
        """
        if not self.landmarks:
            raise ValueError("No landmarks selected. Call select_landmarks() first.")

        with open(output_path, 'w') as f:
            for landmark in self.landmarks:
                f.write(f"{landmark}\n")


# Example usage:
if __name__ == "__main__":
    # Create a simple example graph
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)
    ])
    nx.write_gexf(G, "example_network.gexf")

    # Initialize selector and load network
    selector = constrained_centrality_selection("example_network.gexf")

    # Select landmarks
    landmarks = selector.select_landmarks(num_landmarks=2, h=1)
    print(f"\nSelected landmarks: {landmarks}")

    # Save landmarks to file
    selector.save_landmarks("landmarks.txt")