from collections import defaultdict
import heapq
from typing import Dict, List, Set, Tuple
import json
import os
import sys
import importlib
import networkx as nx


class path_calculator:
    def __init__(self):
        """
        Initialize the path calculator

        Parameters:
        graph: Dict[int, Dict[int, float]] - Graph represented as adjacency list {node: {neighbor: distance}}
        num_landmarks: int - Number of landmarks to select
        """
        self.graph = None
        self.nodes = None
        self.cache_file = None
        self.landmarks = None
        self.distances_to_landmarks = {}
        self.actual_distances = {}

        # Try to load cached distances, if not possible, compute new ones
        # if not self.load_distances():
        #     self.landmarks = self._select_landmarks(num_landmarks)
        #     self.distances_to_landmarks = self._precompute_landmark_distances()
        #     self.save_distances()

    # def _select_landmarks(self, num_landmarks: int) -> List[int]:
    #     """Randomly select specified number of landmarks"""
    #     return random.sample(self.nodes, min(num_landmarks, len(self.nodes)))

    def get_data(self, file_path, landmarks, cache_file):
        self.graph = nx.read_gexf(file_path)
        self.nodes = list(self.graph.nodes())
        self.landmarks = landmarks
        self.cache_file = cache_file
        self.distances_to_landmarks = self._precompute_landmark_distances()
        self.save_distances()

    def _dijkstra(self, start: int) -> Dict[int, float]:
        """Calculate shortest distances from start node to all other nodes using Dijkstra's algorithm"""
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
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

        return distances

    def _precompute_landmark_distances(self) -> Dict[int, Dict[int, float]]:
        """Precompute distances from all nodes to each landmark"""
        distances = {}
        for landmark in self.landmarks:
            distances[landmark] = self._dijkstra(landmark)
        return distances

    def save_distances(self) -> bool:
        """
        Save precomputed distances and landmarks to file
        Returns True if successful, False otherwise
        """
        try:
            # Convert all numeric keys to strings for JSON serialization
            serializable_distances = {}
            for landmark, distances in self.distances_to_landmarks.items():
                serializable_distances[str(landmark)] = {
                    str(node): dist for node, dist in distances.items()
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

            # Verify data structure
            if 'landmarks' not in data or 'distances' not in data:
                return False

            # Keep landmarks as strings
            self.landmarks = data['landmarks']
            self.distances_to_landmarks = {}
            for landmark, distances in data['distances'].items():
                self.distances_to_landmarks[landmark] = distances

            # Verify all nodes are present
            node_set = set(self.nodes)
            for distances in self.distances_to_landmarks.values():
                if set(distances.keys()) != node_set:
                    return False

            return True
        except Exception as e:
            print(f"Error loading distances: {str(e)}")
            return False
    # def load_distances(self, graph, distance_file_path) -> bool:
    #     """
    #     Load precomputed distances and landmarks from file
    #     Returns True if successful, False otherwise
    #     """
    #     try:
    #         self.graph = graph
    #         with open(distance_file_path, 'r') as f:
    #             data = json.load(f)

    #         # Verify data structure
    #         if 'landmarks' not in data or 'distances' not in data:
    #             return False

    #         # Convert string keys back to integers
    #         self.landmarks = data['landmarks']
    #         self.distances_to_landmarks = {}
    #         for landmark_str, distances in data['distances'].items():
    #             landmark = int(landmark_str)
    #             self.distances_to_landmarks[landmark] = {
    #                 int(node): dist for node, dist in distances.items()
    #             }

    #         # Verify all nodes are present
    #         for distances in self.distances_to_landmarks.values():
    #             if set(map(int, distances.keys())) != set(self.nodes):
    #                 return False

    #         return True
    #     except Exception as e:
    #         print(f"Error loading distances: {e}")
    #         return False

    def _estimate_distance(self, source: str, target: str) -> float:
        """
        Estimate distance between two nodes using landmarks
        Using triangle inequality: d(source, target) <= d(source, landmark) + d(landmark, target)
        Returns the minimum upper bound among all landmarks
        """
        if source == target:
            return 0

        estimated = float('inf')
        for landmark in self.landmarks:
            # Get distances, handling possible infinity values
            d_source = self.distances_to_landmarks[landmark].get(source, float('inf'))
            d_target = self.distances_to_landmarks[landmark].get(target, float('inf'))
            
            # Skip if either distance is infinity
            if d_source == float('inf') or d_target == float('inf'):
                continue
                
            dist_via_landmark = d_source + d_target
            estimated = min(estimated, dist_via_landmark)

        return estimated

    # def _estimate_distance(self, source: int, target: int) -> float:
    #     """
    #     Estimate distance between two nodes using landmarks
    #     Using triangle inequality: d(source, target) <= d(source, landmark) + d(landmark, target)
    #     Returns the minimum upper bound among all landmarks
    #     """
    #     if source == target:
    #         return 0

    #     estimated = float('inf')
    #     for landmark in self.landmarks:
    #         dist_via_landmark = (
    #                 self.distances_to_landmarks[landmark][source] +
    #                 self.distances_to_landmarks[landmark][target]
    #         )
    #         estimated = min(estimated, dist_via_landmark)

    #     return estimated

    def _compute_actual_distance(self, source: str, target: str) -> float:
        """Compute actual shortest distance between two nodes"""
        if (source, target) not in self.actual_distances:
            distances = self._dijkstra(source)
            # Handle unreachable nodes
            self.actual_distances[(source, target)] = distances.get(target, float('inf'))
        return self.actual_distances[(source, target)]

    # def _compute_actual_distance(self, source: int, target: int) -> float:
    #     """Compute actual shortest distance between two nodes"""
    #     if (source, target) not in self.actual_distances:
    #         distances = self._dijkstra(source)
    #         self.actual_distances[(source, target)] = distances[target]
    #     return self.actual_distances[(source, target)]

    def compute_error(self, nodes_list) -> float:
        """
        Compute mean error between estimated and actual distance
        Error = |actual - estimated| / actual
        """
        total_error = 0
        count = 0
        for pair in nodes_list:
            source = pair[0]
            target = pair[1]
            
            try:
                if source == target:
                    continue

                actual = self._compute_actual_distance(source, target)
                estimated = self._estimate_distance(source, target)

                # Skip pairs where either distance is infinity
                if actual == float('inf') or estimated == float('inf'):
                    continue
                    
                # Skip if actual distance is 0
                if actual == 0:
                    continue

                error = abs(actual - estimated) / actual
                total_error += error
                count += 1
                
            except Exception as e:
                print(f"Error computing pair {source}-{target}: {str(e)}")
                continue

        if count == 0:
            print("Warning: No valid pairs found for error computation")
            return float('nan')

        return total_error / count

    # def compute_error(self, nodes_list) -> float:
    #     """
    #     Compute mean error between estimated and actual distance
    #     Error = |actual - estimated| / actual
    #     """
    #     total_error = 0
    #     count = 0
    #     for pair in nodes_list:
    #         source = pair[0]
    #         target = pair[1]
    #         if source == target:
    #             continue

    #         actual = self._compute_actual_distance(source, target)
    #         estimated = self._estimate_distance(source, target)

    #         if actual == 0:
    #             if estimated != 0:
    #                 total_error += float('inf')
    #             continue
    #         error = abs(actual - estimated) / actual
    #         total_error += error
    #         count += 1

    #     mean_error = total_error / count
    #     return mean_error


if __name__ == "__main__":
    NUM_LANDMARK = 20
    networks = ["reddit_body", "reddit_title", "wiki", "twitch"]
    selections = {
        # 'basic_selection': {'centrality': "basic_centrality_selection", 'degree': "basic_degree_selection"},
        # 'constrained_selection': {'centrality': "constrained_centrality_selection",
        #                           'degree': "constrained_degree_selection"},
        'partitioning_selection': {'centrality': "partitioning_centrality_selection",
                                  'degree': "partitioning_degree_selection"}
    }
    if 'centrality' in sys.modules:
        del sys.modules['centrality']
    if 'degree' in sys.modules:
        del sys.modules['degree']

    cal_path = path_calculator()
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