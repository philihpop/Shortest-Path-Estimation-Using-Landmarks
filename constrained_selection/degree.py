import networkx as nx
from typing import List, Set
from pathlib import Path


class constrained_degree_selection:
    """
    A class to select landmarks in networks using degree-based constrained heuristic approach.
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
        Select landmarks iteratively based on node degrees.
        For each selected landmark l, remove all nodes at distance h or less from l.

        Args:
            num_landmarks: Number of landmarks to select
            h: Distance parameter (nodes within distance h from a landmark are removed)

        Returns:
            List of selected landmark node IDs
        """
        if not self.G:
            raise ValueError("No network loaded. Call load_network() first.")

        # Get initial degree-based ranking
        degree_ranking = sorted(self.G.degree(), key=lambda x: x[1], reverse=True)
        ranked_nodes = [node for node, _ in degree_ranking]

        self.landmarks = []
        available_nodes = set(self.G.nodes())

        while len(self.landmarks) < num_landmarks and available_nodes:
            # Select highest-degree available node as new landmark
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

    # Initialize selector and load network
    selector = constrained_degree_selection("example_network.gexf")

    # Select landmarks
    landmarks = selector.select_landmarks(num_landmarks=2, h=1)
    print(f"\nSelected landmarks: {landmarks}")

    # Save landmarks to file
    # selector.save_landmarks("landmarks.txt")
