from collections import defaultdict
import heapq
from typing import Dict, List, Set, Tuple
import json
import os
import sys
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from functools import partial
import importlib

class parallel_path_calculator:
    def __init__(self, num_workers: int = 24):
        self.graph = None
        self.nodes = None
        self.cache_file = None
        self.landmarks = None
        self.distances_to_landmarks = {}
        self.actual_distances = {}
        self.num_workers = num_workers

    def get_data(self, file_path, landmarks, cache_file):
        self.graph = nx.read_gexf(file_path)
        self.nodes = list(self.graph.nodes())
        self.landmarks = landmarks
        self.cache_file = cache_file
        self.distances_to_landmarks = self._parallel_precompute_landmark_distances()
        self.save_distances()

    def _parallel_dijkstra_worker(self, start_node: str) -> Tuple[str, Dict[str, float]]:
        """Worker function for parallel Dijkstra computation"""
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        pq = [(0, start_node)]
        visited = set()

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor, _ in self.graph[current_node].items():
                distance = current_distance + 1

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return start_node, distances

    def _parallel_precompute_landmark_distances(self) -> Dict[str, Dict[str, float]]:
        """Precompute distances from all landmarks in parallel"""
        distances = {}
        print(f"Computing distances from {len(self.landmarks)} landmarks in parallel...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._parallel_dijkstra_worker, landmark) 
                      for landmark in self.landmarks]
            
            for future in tqdm(futures):
                landmark, landmark_distances = future.result()
                distances[landmark] = landmark_distances

        return distances

    def _parallel_compute_errors_chunk(self, node_pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute errors for a chunk of node pairs"""
        errors = []
        for source, target in node_pairs:
            if source == target:
                continue

            actual = self._compute_actual_distance(source, target)
            estimated = self._estimate_distance(source, target)

            if actual == float('inf') or estimated == float('inf') or actual == 0:
                continue

            error = abs(actual - estimated) / actual
            errors.append(error)
        
        return errors

    def save_distances(self) -> bool:
        try:
            serializable_distances = {
                str(landmark): {str(node): dist for node, dist in distances.items()}
                for landmark, distances in self.distances_to_landmarks.items()
            }

            data = {
                'landmarks': self.landmarks,
                'distances': serializable_distances
            }

            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving distances: {e}")
            return False

    def load_distances(self, graph, distance_file_path) -> bool:
        try:
            self.graph = graph
            self.nodes = list(self.graph.nodes())

            if not os.path.exists(distance_file_path):
                print(f"Error loading distances: File not found - {distance_file_path}")
                return False

            with open(distance_file_path, 'r') as f:
                data = json.load(f)

            if 'landmarks' not in data or 'distances' not in data:
                return False

            self.landmarks = data['landmarks']
            self.distances_to_landmarks = data['distances']

            node_set = set(self.nodes)
            for distances in self.distances_to_landmarks.values():
                if set(distances.keys()) != node_set:
                    return False

            return True
        except Exception as e:
            print(f"Error loading distances: {str(e)}")
            return False

    def _estimate_distance(self, source: str, target: str) -> float:
        if source == target:
            return 0

        estimated = float('inf')
        for landmark in self.landmarks:
            d_source = float(self.distances_to_landmarks[landmark].get(source, float('inf')))
            d_target = float(self.distances_to_landmarks[landmark].get(target, float('inf')))
            
            if d_source == float('inf') or d_target == float('inf'):
                continue
                
            dist_via_landmark = d_source + d_target
            estimated = min(estimated, dist_via_landmark)

        return estimated

    def _compute_actual_distance(self, source: str, target: str) -> float:
        if (source, target) not in self.actual_distances:
            _, distances = self._parallel_dijkstra_worker(source)
            self.actual_distances[(source, target)] = distances.get(target, float('inf'))
        return self.actual_distances[(source, target)]

    def compute_error(self, nodes_list) -> float:
        """Compute mean error using parallel processing"""
        chunk_size = max(1, len(nodes_list) // (self.num_workers * 4))  # Smaller chunks for better load balancing
        node_chunks = [nodes_list[i:i + chunk_size] 
                      for i in range(0, len(nodes_list), chunk_size)]

        print(f"Computing errors for {len(nodes_list)} pairs in parallel...")
        total_error = 0
        total_count = 0

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._parallel_compute_errors_chunk, chunk) 
                      for chunk in node_chunks]
            
            for future in tqdm(futures):
                chunk_errors = future.result()
                total_error += sum(chunk_errors)
                total_count += len(chunk_errors)

        if total_count == 0:
            print("Warning: No valid pairs found for error computation")
            return float('nan')

        return total_error / total_count

if __name__ == "__main__":
    NUM_LANDMARK = 100
    networks = ["reddit_body", "reddit_title", "wiki", "twitch", "CA"]
    # networks = [ "wiki", "twitch", "CA"]
    selections = {
        # 'basic_selection': {'degree': "basic_degree_selection"},
        'constrained_selection': {'centrality': "constrained_centrality_selection",
                                  'degree': "constrained_degree_selection"},
        # 'partitioning_selection': {'centrality': "partitioning_centrality_selection",
        #                           'degree': "partitioning_degree_selection"}
    }
    
    cal_path = parallel_path_calculator(num_workers=24)  # Use 24 cores
    for name, selection_dic in selections.items():
        module_dir = os.path.join(os.path.dirname(__file__), name)
        print(f"Loading from directory: {module_dir}")  # Debug print
        
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)  # Insert at beginning of path

        for module_name, selection in selection_dic.items():
            for network in networks:
                try:
                    print(f"Importing {module_name} from {module_dir}")  # Debug print
                    module = importlib.import_module(module_name)
                    print(f"Available in module: {dir(module)}")  # Debug print
                    selector_class = getattr(module, selection)
                    in_file_path = "data to graph/" + network + ".gexf"
                    out_file_path = name + "/pre_compute distance/" + selection + "_" + network + ".json"
                    selector = selector_class(gexf_path=in_file_path)
                    landmark_list = selector.select_landmarks(NUM_LANDMARK)
                    cal_path.get_data(in_file_path, landmark_list, out_file_path)
                except Exception as e:
                    print(f"Error processing {network} with {module_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()  # Print full error trace