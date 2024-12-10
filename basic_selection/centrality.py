import networkx as nx
import numpy as np
from typing import List, Dict, Set, Optional
import random
from collections import defaultdict
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

class basic_centrality_selection:
    """
    An improved class for selecting landmarks based on approximate closeness centrality
    using stratified sampling and parallel processing.
    """

    def __init__(
            self,
            gexf_path: str,
            num_seeds: int = 25,
            random_seed: Optional[int] = 42,
            num_workers: int = 24
    ):
        """
        Initialize the ImprovedClosenessSelection.

        Parameters:
            gexf_path (str): Path to the GEXF file
            num_seeds (int): Number of random seeds for approximation
            random_seed (Optional[int]): Random seed for reproducibility
            num_workers (int): Number of parallel workers
        """
        self.graph = None
        self.num_landmarks = None
        self.num_seeds = num_seeds
        self.random_seed = random_seed
        self.num_workers = num_workers
        self.landmarks = None
        self.centrality_scores = None
        self.load_network(gexf_path)

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def load_network(self, file_path: str) -> None:
        """Load network from a GEXF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Network file not found: {file_path}")

        try:
            self.graph = nx.read_gexf(file_path)
            print(f"Successfully loaded network with {self.graph.number_of_nodes()} nodes "
                  f"and {self.graph.number_of_edges()} edges")
        except Exception as e:
            raise nx.NetworkXError(f"Error loading GEXF file: {str(e)}")

    def _get_degree_strata(self) -> Dict[str, int]:
        """
        Divide nodes into strata based on degree for stratified sampling.
        
        Returns:
            Dict[str, int]: Dictionary mapping node IDs to their strata
        """
        degrees = dict(self.graph.degree())
        degree_thresholds = np.percentile(list(degrees.values()), [25, 50, 75])
        
        strata = {}
        for node, degree in degrees.items():
            if degree <= degree_thresholds[0]:
                strata[node] = 0
            elif degree <= degree_thresholds[1]:
                strata[node] = 1
            elif degree <= degree_thresholds[2]:
                strata[node] = 2
            else:
                strata[node] = 3
                
        return strata

    def _parallel_bfs_worker(self, seed_batch: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Parallel worker function to compute BFS distances for a batch of seeds.
        
        Parameters:
            seed_batch (List[str]): Batch of seed nodes
            
        Returns:
            Dict[str, Dict[str, int]]: Distances from each seed to reachable nodes
        """
        results = {}
        for seed in seed_batch:
            distances = {seed: 0}
            queue = [(seed, 0)]
            index = 0

            while index < len(queue):
                current, dist = queue[index]
                index += 1

                for neighbor in self.graph.neighbors(current):
                    if neighbor not in distances:
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            
            results[seed] = distances
        return results

    def compute_approximate_closeness(self) -> Dict[str, float]:
        """
        Compute approximate closeness centrality using stratified sampling
        and parallel processing.
        """
        if self.graph is None:
            raise RuntimeError("No network loaded. Call load_network() first.")

        # Get degree-based strata
        strata = self._get_degree_strata()
        strata_nodes = defaultdict(list)
        for node, stratum in strata.items():
            strata_nodes[stratum].append(node)

        # Stratified sampling
        seeds_per_stratum = math.ceil(self.num_seeds / 4)  # Divide seeds among strata
        seed_nodes = []
        for stratum in range(4):
            stratum_size = len(strata_nodes[stratum])
            n_seeds = min(seeds_per_stratum, stratum_size)
            if n_seeds > 0:
                seed_nodes.extend(random.sample(strata_nodes[stratum], n_seeds))

        # Prepare batches for parallel processing
        batch_size = max(1, len(seed_nodes) // self.num_workers)
        seed_batches = [
            seed_nodes[i:i + batch_size]
            for i in range(0, len(seed_nodes), batch_size)
        ]

        # Parallel BFS computation
        total_distances = defaultdict(int)
        reachable_count = defaultdict(int)
        
        print("Computing approximate closeness centrality in parallel...")
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._parallel_bfs_worker, batch)
                for batch in seed_batches
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                batch_results = future.result()
                for seed_results in batch_results.values():
                    for node, dist in seed_results.items():
                        total_distances[node] += dist
                        reachable_count[node] += 1

        # Compute weighted centrality scores
        self.centrality_scores = {}
        for node in self.graph.nodes():
            if reachable_count[node] > 0:
                avg_distance = total_distances[node] / reachable_count[node]
                # Weight by degree to favor high-degree nodes
                degree_weight = math.log2(self.graph.degree(node) + 1)
                self.centrality_scores[node] = (1.0 / avg_distance) * degree_weight if avg_distance > 0 else 0.0
            else:
                self.centrality_scores[node] = 0.0

        return self.centrality_scores

    def select_landmarks(self, num: int) -> List[str]:
        """Select landmarks based on improved approximate closeness centrality."""
        self.num_landmarks = num
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