import networkx as nx
import numpy as np
from typing import List, Dict, Set, Optional
import random
from collections import defaultdict
import os
from tqdm import tqdm


class basic_centrality_selection:
    """
    A class for selecting landmarks based on approximate closeness centrality
    using random seed sampling and BFS.

    Attributes:
        graph (nx.Graph): The network graph
        num_landmarks (int): Number of landmarks to select
        num_seeds (int): Number of random seeds for approximation
        random_seed (int): Random seed for reproducibility
        landmarks (List[str]): Selected landmark nodes
        centrality_scores (Dict[str, float]): Approximate closeness centrality scores
    """

    def __init__(
            self,
            num_landmarks: int,
            num_seeds: int = 100,
            random_seed: Optional[int] = None
    ):
        """
        Initialize the ApproxClosenessSelection.

        Parameters:
            num_landmarks (int): Number of landmarks to select
            num_seeds (int): Number of random seeds for approximation
            random_seed (Optional[int]): Random seed for reproducibility
        """
        self.graph = None
        self.num_landmarks = num_landmarks
        self.num_seeds = num_seeds
        self.random_seed = random_seed
        self.landmarks = None
        self.centrality_scores = None

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def load_network(self, file_path: str) -> None:
        """
        Load network from a GEXF file.

        Parameters:
            file_path (str): Path to the .gexf file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Network file not found: {file_path}")

        try:
            self.graph = nx.read_gexf(file_path)
            print(f"Successfully loaded network with {self.graph.number_of_nodes()} nodes "
                  f"and {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise nx.NetworkXError(f"Error loading GEXF file: {str(e)}")

    def _bfs_distances(self, source: str) -> Dict[str, int]:
        """
        Perform BFS from a source node and return distances to all reachable nodes.

        Parameters:
            source (str): Source node for BFS

        Returns:
            Dict[str, int]: Dictionary of node distances from source
        """
        distances = {source: 0}
        queue = [(source, 0)]
        index = 0

        while index < len(queue):
            current, dist = queue[index]
            index += 1

            for neighbor in self.graph.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

        return distances

    def compute_approximate_closeness(self) -> Dict[str, float]:
        """
        Compute approximate closeness centrality using random seed sampling.

        Returns:
            Dict[str, float]: Dictionary of approximate closeness centrality scores
        """
        if self.graph is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)

        # Select random seed nodes
        seed_nodes = random.sample(nodes, min(self.num_seeds, n_nodes))

        # Initialize sum of distances for each node
        total_distances = defaultdict(int)
        reachable_count = defaultdict(int)

        # Perform BFS from each seed node
        print("Computing approximate closeness centrality...")
        for seed in tqdm(seed_nodes):
            distances = self._bfs_distances(seed)

            # Update total distances and reachable counts
            for node, dist in distances.items():
                total_distances[node] += dist
                reachable_count[node] += 1

        # Compute approximate closeness centrality
        self.centrality_scores = {}
        for node in nodes:
            if reachable_count[node] > 0:
                # Average distance to seed nodes
                avg_distance = total_distances[node] / reachable_count[node]
                # Approximate closeness centrality
                self.centrality_scores[node] = 1.0 / avg_distance if avg_distance > 0 else 0.0
            else:
                self.centrality_scores[node] = 0.0

        return self.centrality_scores

    def select_landmarks(self) -> List[str]:
        """
        Select landmarks based on approximate closeness centrality.
        Selects nodes with lowest centrality scores as landmarks.

        Returns:
            List[str]: List of selected landmark nodes
        """
        if self.centrality_scores is None:
            self.compute_approximate_closeness()

        # Sort nodes by centrality score in ascending order (lowest first)
        sorted_nodes = sorted(
            self.centrality_scores.items(),
            key=lambda x: x[1]
        )

        # Select nodes with lowest centrality scores as landmarks
        self.landmarks = [node for node, score in sorted_nodes[:self.num_landmarks]]

        return self.landmarks

    def analyze_coverage(self) -> dict:
        """
        Analyze the coverage properties of selected landmarks.

        Returns:
            dict: Dictionary containing coverage statistics
        """
        if self.landmarks is None:
            raise RuntimeError("No landmarks selected. Call select_landmarks() first.")

        # Compute average distance from landmarks to all other nodes
        total_distances = defaultdict(int)
        reachable_count = defaultdict(int)

        for landmark in self.landmarks:
            distances = self._bfs_distances(landmark)
            for node, dist in distances.items():
                total_distances[node] += dist
                reachable_count[node] += 1

        # Calculate coverage metrics
        n_nodes = self.graph.number_of_nodes()
        covered_nodes = sum(1 for count in reachable_count.values() if count > 0)
        avg_distance = np.mean([
            total_distances[node] / reachable_count[node]
            for node in total_distances
            if reachable_count[node] > 0
        ])

        coverage_stats = {
            'num_landmarks': len(self.landmarks),
            'coverage_ratio': covered_nodes / n_nodes,
            'average_distance': avg_distance,
            'landmark_centralities': {
                node: self.centrality_scores[node]
                for node in self.landmarks
            }
        }

        return coverage_stats

    def save_results(self, output_file: str, coverage_stats: dict) -> None:
        """
        Save landmarks and their statistics to a file.

        Parameters:
            output_file (str): Path to save the results
            coverage_stats (dict): Coverage statistics dictionary
        """
        with open(output_file, 'w') as f:
            f.write("Selected Landmarks:\n")
            for i, landmark in enumerate(self.landmarks, 1):
                centrality = self.centrality_scores[landmark]
                f.write(f"{i}. Node {landmark} "
                        f"(approx. closeness centrality: {centrality:.6f})\n")

            f.write("\nCoverage Statistics:\n")
            for metric, value in coverage_stats.items():
                if metric != 'landmark_centralities':
                    f.write(f"{metric}: {value:.6f}\n" if isinstance(value, float)
                            else f"{metric}: {value}\n")


# Example usage
if __name__ == "__main__":
    try:
        # Initialize selector
        selector = basic_centrality_selection(
            num_landmarks=5,
            num_seeds=100,
            random_seed=42
        )

        # Load network and select landmarks
        selector.load_network("../data to graph/reddit_body.gexf")
        landmarks = selector.select_landmarks()

        # Analyze coverage
        stats = selector.analyze_coverage()

        # Save results
        # selector.save_results("landmarks_closeness.txt", stats)

        # Print summary
        print(f"\nSelected {len(landmarks)} landmarks")
        print(f"Coverage ratio: {stats['coverage_ratio']:.2%}")
        print(f"Average distance: {stats['average_distance']:.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")