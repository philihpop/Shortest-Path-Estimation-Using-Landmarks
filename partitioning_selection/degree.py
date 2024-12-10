import networkx as nx
from typing import List, Dict, Set, Tuple
from pathlib import Path
from tqdm import tqdm
from networkx.algorithms.community import kernighan_lin_bisection
from concurrent.futures import ProcessPoolExecutor
import math
from collections import defaultdict

class partitioning_degree_selection:
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

    def _parallel_bisect(self, data: Tuple[nx.Graph, int]) -> Tuple[int, List, List]:
        """Parallel worker for graph bisection."""
        subgraph, partition_id = data
        try:
            sets = kernighan_lin_bisection(subgraph)
            if sets:
                set1, set2 = sets
                return partition_id, list(set1), list(set2)
        except:
            pass
        return partition_id, None, None

    def _compute_degrees_chunk(self, nodes: List[str]) -> Dict[str, int]:
        """Compute degrees for a chunk of nodes."""
        return {node: self.G.degree(node) for node in nodes}

    def _chunk_nodes(self, nodes: List[str], num_chunks: int) -> List[List[str]]:
        """Split nodes into chunks for parallel processing."""
        chunk_size = math.ceil(len(nodes) / num_chunks)
        return [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]

    def select_landmarks(self, num_landmarks: int) -> List[str]:
        if not self.G:
            raise ValueError("No network loaded. Call load_network() first.")

        print("Partitioning graph...")
        G_undirected = self.G.to_undirected()
        partitions = {node: 0 for node in self.G.nodes()}
        partition_to_nodes = {0: list(self.G.nodes())}
        current_partitions = 1
        target_partitions = num_landmarks
        
        with tqdm(total=target_partitions-1, desc="Creating partitions") as pbar:
            while current_partitions < target_partitions:
                # Find partitions to process in parallel
                partitions_to_process = []
                for partition_id in list(partition_to_nodes.keys()):
                    if len(partition_to_nodes[partition_id]) > 1:
                        subgraph = G_undirected.subgraph(partition_to_nodes[partition_id])
                        partitions_to_process.append((subgraph, partition_id))
                        if len(partitions_to_process) >= self.num_workers:
                            break
                
                if not partitions_to_process:
                    break
                
                # Process partitions in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(self._parallel_bisect, data) 
                             for data in partitions_to_process]
                    
                    for future in futures:
                        partition_id, set1, set2 = future.result()
                        if set1 and set2:
                            # Update partitions
                            del partition_to_nodes[partition_id]
                            new_partition1 = current_partitions
                            new_partition2 = current_partitions + 1
                            
                            partition_to_nodes[new_partition1] = set1
                            partition_to_nodes[new_partition2] = set2
                            
                            for node in set1:
                                partitions[node] = new_partition1
                            for node in set2:
                                partitions[node] = new_partition2
                                
                            current_partitions += 2
                            pbar.update(1)

        print(f"Created {len(partition_to_nodes)} partitions")

        # Compute degrees in parallel
        print("Computing node degrees in parallel...")
        nodes = list(self.G.nodes())
        node_chunks = self._chunk_nodes(nodes, self.num_workers)
        degrees = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for chunk in node_chunks:
                futures.append(executor.submit(self._compute_degrees_chunk, chunk))

            with tqdm(total=len(futures), desc="Computing degrees") as pbar:
                for future in futures:
                    chunk_degrees = future.result()
                    degrees.update(chunk_degrees)
                    pbar.update(1)

        # Select landmarks from partitions
        landmarks_per_partition = max(1, num_landmarks // len(partition_to_nodes))
        self.landmarks = []
        
        print(f"Selecting {landmarks_per_partition} landmarks from each partition...")
        with tqdm(total=num_landmarks, desc="Selecting landmarks") as pbar:
            for partition_id in sorted(partition_to_nodes.keys()):
                partition_nodes = partition_to_nodes[partition_id]
                # Sort nodes in partition by degree
                partition_nodes.sort(key=lambda x: degrees[x], reverse=True)
                
                # Select top nodes from partition
                selected = partition_nodes[:landmarks_per_partition]
                self.landmarks.extend(selected)
                pbar.update(len(selected))
                
                if len(self.landmarks) >= num_landmarks:
                    break

        return self.landmarks[:num_landmarks]

if __name__ == "__main__":
    # Initialize selector with parallel processing
    selector = parallel_partitioning_degree_selection(
        "example_network.gexf",
        num_workers=24
    )
    
    # Select landmarks
    landmarks = selector.select_landmarks(num_landmarks=100)
    print(f"\nSelected landmarks: {landmarks}")