import networkx as nx
import random
from cal_distance import path_calculator


def get_random_node_pairs(graph, num_pairs):
    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("The graph must contain at least two nodes")
    node_pairs = set()
    while len(node_pairs) < num_pairs:
        pair = tuple(random.sample(nodes, 2))
        if pair not in node_pairs:
            node_pairs.add(pair)
    return list(node_pairs)


networks = ["reddit_body", "reddit_title"]
selections = {
    'basic_selection': ["basic_centrality_selection", "basic_degree_selection"],
    # 'constrained_selection': {'centrality': "constrained_centrality_selection",
    #                           'degree': "constrained_degree_selection"},
    # 'partitioning_selection': {'centrality': "partitioning_centrality_selection",
    #                            'degree': "partitioning_degree_selection"}
}
cal_path = path_calculator()
for network in networks:
    graph_file_path = "data to graph/" + network + ".gexf"
    graph = nx.read_gexf(graph_file_path)
    # select 100 pair nodes randomly
    random_pairs = get_random_node_pairs(graph, 100)
    print("-------" + network + "-----------")
    for name, selection_list in selections.items():
        for selection in selection_list:
            dis_file_path = name + "/pre_compute distance/" + selection + "_" + network + ".json"
            if cal_path.load_distances(graph, dis_file_path):
                error = cal_path.compute_error(random_pairs)
                print(selection + ": " + str(error))
