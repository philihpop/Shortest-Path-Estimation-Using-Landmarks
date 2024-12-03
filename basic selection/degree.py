import networkx as nx
from typing import List, Union, Optional
import numpy as np
import os


class basic_degree_selection:
    """
    A class for selecting landmark nodes from a network based on node degrees.

    Attributes:
        graph (nx.Graph): The network graph
        num_landmarks (int): Number of landmarks to select
        weighted (bool): Whether to use weighted degrees
        weight_attr (str): Name of the weight attribute for weighted graphs
        landmarks (List[str]): Selected landmark nodes
        coverage_stats (dict): Statistics about landmark coverage
    """

    def __init__(
            self,
            num_landmarks: int,
            weighted: bool = False,
            weight_attr: str = 'weight'
    ):
        """
        Initialize the BasicDegreeSelection.

        Parameters:
            num_landmarks (int): Number of landmarks to select
            weighted (bool): Whether to use weighted degrees
            weight_attr (str): Name of the weight attribute for weighted graphs
        """
        self.graph = None
        self.num_landmarks = num_landmarks
        self.weighted = weighted
        self.weight_attr = weight_attr
        self.landmarks = None
        self.coverage_stats = None

    def load_network(self, file_path: str) -> None:
        """
        Load network from a GEXF file.

        Parameters:
            file_path (str): Path to the .gexf file

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            nx.NetworkXError: If there's an error parsing the GEXF file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Network file not found: {file_path}")

        try:
            self.graph = nx.read_gexf(file_path)
            print(f"Successfully loaded network with {self.graph.number_of_nodes()} nodes "
                  f"and {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise nx.NetworkXError(f"Error loading GEXF file: {str(e)}")

    def select_landmarks(self) -> List[str]:
        """
        Select landmarks from the network based on node degrees.

        Returns:
            List[str]: List of node IDs selected as landmarks

        Raises:
            ValueError: If num_landmarks is greater than the number of nodes
            RuntimeError: If network hasn't been loaded
        """
        if self.graph is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        if self.num_landmarks > self.graph.number_of_nodes():
            raise ValueError(
                f"Number of requested landmarks ({self.num_landmarks}) exceeds "
                f"number of nodes in graph ({self.graph.number_of_nodes()})"
            )

        # Calculate node degrees
        if self.weighted and nx.is_weighted(self.graph, weight=self.weight_attr):
            degrees = dict(nx.degree(self.graph, weight=self.weight_attr))
        else:
            degrees = dict(nx.degree(self.graph))

        # Sort nodes by degree in descending order
        sorted_nodes = sorted(
            degrees.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top D nodes as landmarks
        self.landmarks = [node for node, degree in sorted_nodes[:self.num_landmarks]]

        return self.landmarks

    def analyze_coverage(self) -> dict:
        """
        Analyze the coverage properties of selected landmarks.

        Returns:
            dict: Dictionary containing coverage statistics

        Raises:
            RuntimeError: If landmarks haven't been selected yet
        """
        if self.landmarks is None:
            raise RuntimeError("No landmarks selected. Call select_landmarks() first.")

        total_nodes = self.graph.number_of_nodes()
        landmark_neighbors = set()

        # Get all unique neighbors of landmarks
        for landmark in self.landmarks:
            landmark_neighbors.update(self.graph.neighbors(landmark))

        # Remove landmarks from neighbor count
        landmark_neighbors = landmark_neighbors - set(self.landmarks)

        # Calculate coverage metrics
        self.coverage_stats = {
            'num_landmarks': len(self.landmarks),
            'num_neighbors': len(landmark_neighbors),
            'coverage_ratio': len(landmark_neighbors) / total_nodes,
            'average_degree': np.mean([self.graph.degree(node) for node in self.landmarks]),
            'landmark_degrees': {node: self.graph.degree(node) for node in self.landmarks}
        }

        return self.coverage_stats

    def save_results(self, output_file: str) -> None:
        """
        Save landmarks and their statistics to a file.

        Parameters:
            output_file (str): Path to save the results

        Raises:
            RuntimeError: If analysis hasn't been performed yet
        """
        if self.coverage_stats is None:
            raise RuntimeError("No analysis results. Call analyze_coverage() first.")

        with open(output_file, 'w') as f:
            f.write("Selected Landmarks:\n")
            for i, landmark in enumerate(self.landmarks, 1):
                degree = self.coverage_stats['landmark_degrees'][landmark]
                f.write(f"{i}. Node {landmark} (degree: {degree})\n")

            f.write("\nCoverage Statistics:\n")
            for metric, value in self.coverage_stats.items():
                if metric != 'landmark_degrees':
                    f.write(f"{metric}: {value:.3f}" if isinstance(value, float)
                            else f"{metric}: {value}\n")

    def get_landmarks(self) -> List[str]:
        """
        Get the list of selected landmarks.

        Returns:
            List[str]: List of landmark node IDs
        """
        return self.landmarks

    def get_coverage_stats(self) -> dict:
        """
        Get the coverage statistics.

        Returns:
            dict: Coverage statistics dictionary
        """
        return self.coverage_stats


# Example usage
if __name__ == "__main__":
    try:
        # Initialize selector
        selector = basic_degree_selection(num_landmarks=5, weighted=False)

        # Load network
        selector.load_network("../data to graph/reddit_body.gexf")

        # Select landmarks
        landmarks = selector.select_landmarks()

        # Analyze coverage
        stats = selector.analyze_coverage()

        # Save results
        # selector.save_results("landmarks.txt")

        # Print summary
        print(f"\nSelected {len(landmarks)} landmarks")
        print(f"Coverage ratio: {stats['coverage_ratio']:.2%}")

    except Exception as e:
        print(f"Error: {str(e)}")