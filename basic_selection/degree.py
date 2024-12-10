import networkx as nx
from typing import List, Dict, Tuple
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
from itertools import islice

class basic_degree_selection:
    """
    A class for selecting landmark nodes from a network based on node degrees using parallel processing.
    """
    def __init__(
            self,
            gexf_path: str,
            weighted: bool = False,
            weight_attr: str = 'weight',
            num_workers: int = 24
    ):
        self.graph = None
        self.num_landmarks = None
        self.weighted = weighted
        self.weight_attr = weight_attr
        self.landmarks = None
        self.coverage_stats = None
        self.num_workers = num_workers
        self.load_network(gexf_path)

    def load_network(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Network file not found: {file_path}")

        try:
            self.graph = nx.read_gexf(file_path)
            print(f"Successfully loaded network with {self.graph.number_of_nodes()} nodes "
                  f"and {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise nx.NetworkXError(f"Error loading GEXF file: {str(e)}")

    def _calculate_degrees_chunk(self, nodes_chunk: List[str]) -> List[Tuple[str, float]]:
        """
        Calculate degrees for a chunk of nodes.
        
        Parameters:
            nodes_chunk: List of nodes to process
            
        Returns:
            List of (node, degree) tuples
        """
        result = []
        for node in nodes_chunk:
            if self.weighted and nx.is_weighted(self.graph, weight=self.weight_attr):
                degree = sum(w.get(self.weight_attr, 1.0) 
                           for _, _, w in self.graph.edges(node, data=True))
            else:
                degree = self.graph.degree(node)
            result.append((node, degree))
        return result

    def _chunks(self, lst: List, n: int):
        """Split list into n roughly equal chunks."""
        chunk_size = math.ceil(len(lst) / n)
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def select_landmarks(self, num: int) -> List[str]:
        """Select landmarks using parallel degree computation."""
        self.num_landmarks = num
        if self.graph is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        if self.num_landmarks > self.graph.number_of_nodes():
            raise ValueError(
                f"Number of requested landmarks ({self.num_landmarks}) exceeds "
                f"number of nodes in graph ({self.graph.number_of_nodes()})"
            )

        print("Computing node degrees in parallel...")
        nodes = list(self.graph.nodes())
        
        # Split nodes into chunks for parallel processing
        node_chunks = list(self._chunks(nodes, self.num_workers))
        
        # Process chunks in parallel
        degrees = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Map chunks to worker processes
            future_to_chunk = {
                executor.submit(self._calculate_degrees_chunk, chunk): i 
                for i, chunk in enumerate(node_chunks)
            }
            
            # Collect results with progress bar
            for future in tqdm(future_to_chunk, total=len(node_chunks)):
                chunk_results = future.result()
                degrees.extend(chunk_results)

        print("Sorting nodes by degree...")
        # Sort nodes by degree in descending order
        sorted_nodes = sorted(
            degrees,
            key=lambda x: x[1],
            reverse=True
        )

        # Select top D nodes as landmarks
        self.landmarks = [node for node, degree in sorted_nodes[:self.num_landmarks]]
        print(f"Selected {len(self.landmarks)} landmarks based on degree centrality")

        return self.landmarks

    def get_landmarks(self) -> List[str]:
        """Get the list of selected landmarks."""
        return self.landmarks