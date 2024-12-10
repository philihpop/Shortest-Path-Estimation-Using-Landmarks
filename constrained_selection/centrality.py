import networkx as nx
from typing import List, Set, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import defaultdict
import math

class constrained_centrality_selection:
    def __init__(self, gexf_path: str = None, num_workers: int = 24):
        self.G = None
        self.landmarks = []
        self.num_workers = num_workers
        if gexf_path:
            self.load_network(gexf_path)

    def load_network(self, network_path: str) -> None:
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

    def _compute_centrality_chunk(self, data: Tuple[List[str], List[str]]) -> Dict[str, float]:
        """Compute centrality for a chunk of source nodes."""
        source_nodes, target_nodes = data
        centrality_scores = defaultdict(float)
        
        for source in source_nodes:
            try:
                distances = nx.single_source_shortest_path_length(self.G, source)
                for node in target_nodes:
                    if node in distances and distances[node] > 0:
                        centrality_scores[node] += 1.0 / distances[node]
            except nx.NetworkXError:
                continue
                
        return dict(centrality_scores)

    def _compute_removal_set(self, data: Tuple[str, int, Set[str]]) -> Set[str]:
        """Compute nodes to remove for a given landmark."""
        landmark, h, available_nodes = data
        to_remove = set()
        
        try:
            paths = nx.single_source_shortest_path_length(self.G, landmark)
            for node, distance in paths.items():
                if distance <= h and node in available_nodes:
                    to_remove.add(node)
        except nx.NetworkXError:
            pass
            
        return to_remove

    def _chunk_nodes(self, nodes: List[str], num_chunks: int) -> List[List[str]]:
        """Split nodes into chunks for parallel processing."""
        chunk_size = math.ceil(len(nodes) / num_chunks)
        return [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]

    def select_landmarks(self, num_landmarks: int, h: int = 1) -> List[str]:
        if not self.G:
            raise ValueError("No network loaded. Call load_network() first.")

        print("Computing approximate closeness centrality in parallel...")
        nodes = list(self.G.nodes())
        num_samples = min(100, len(nodes))
        sample_nodes = random.sample(nodes, num_samples)

        # Split computation into chunks for parallel processing
        source_chunks = self._chunk_nodes(sample_nodes, self.num_workers)
        centrality_scores = defaultdict(float)

        # Compute centrality scores in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for source_chunk in source_chunks:
                futures.append(
                    executor.submit(
                        self._compute_centrality_chunk,
                        (source_chunk, nodes)
                    )
                )

            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Computing centrality") as pbar:
                for future in futures:
                    chunk_scores = future.result()
                    for node, score in chunk_scores.items():
                        centrality_scores[node] += score
                    pbar.update(1)

        # Normalize scores
        max_score = max(centrality_scores.values()) if centrality_scores else 1.0
        if max_score > 0:
            centrality_scores = {
                node: score/max_score 
                for node, score in centrality_scores.items()
            }

        print("Sorting nodes by centrality...")
        ranked_nodes = sorted(centrality_scores.keys(),
                            key=lambda k: centrality_scores[k])

        self.landmarks = []
        available_nodes = set(self.G.nodes())

        print(f"Selecting {num_landmarks} landmarks in parallel...")
        with tqdm(total=num_landmarks) as pbar:
            while len(self.landmarks) < num_landmarks and available_nodes:
                # Select next batch of candidate landmarks
                batch_size = min(self.num_workers, num_landmarks - len(self.landmarks))
                candidates = []
                
                for node in ranked_nodes:
                    if node in available_nodes:
                        candidates.append(node)
                        if len(candidates) == batch_size:
                            break
                
                if not candidates:
                    break

                # Compute removal sets in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = []
                    for landmark in candidates:
                        futures.append(
                            executor.submit(
                                self._compute_removal_set,
                                (landmark, h, available_nodes)
                            )
                        )

                    # Process results and update available nodes
                    for landmark, future in zip(candidates, futures):
                        to_remove = future.result()
                        self.landmarks.append(landmark)
                        available_nodes -= to_remove
                        print(f"Selected landmark {landmark} with centrality: "
                              f"{centrality_scores[landmark]:.4f}")
                        print(f"Removed {len(to_remove)} nodes within distance {h}")
                        pbar.update(1)

        return self.landmarks

    def save_landmarks(self, output_path: str) -> None:
        if not self.landmarks:
            raise ValueError("No landmarks selected. Call select_landmarks() first.")

        with open(output_path, 'w') as f:
            for landmark in self.landmarks:
                f.write(f"{landmark}\n")

if __name__ == "__main__":
    # Initialize selector with parallel processing
    selector = parallel_constrained_centrality_selection(
        "example_network.gexf",
        num_workers=24
    )
    
    # Select landmarks
    landmarks = selector.select_landmarks(num_landmarks=100, h=1)
    print(f"\nSelected landmarks: {landmarks}")