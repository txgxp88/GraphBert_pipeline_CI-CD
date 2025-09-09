#######################
###  WLGraphColoring
#######################

# Initial Labeling:
# Each node is assigned an initial label.
# This could be a node feature, degree, or even a default constant (e.g., all nodes start with "1").
# Iterative Relabeling:
# At each iteration:
# For each node, collect its own label and the labels of its neighbors.
# Concatenate the current label with the sorted list of neighbor labels.
# Hash this concatenated string to get a new label.
# Repeat this process for a fixed number of iterations or until convergence.
# Final Output:
# The final node labels after k iterations encode structural information about the graph.
# These can be used as node features or for graph comparison.



import hashlib
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

class WLGraphColoring:
    def __init__(self, max_iter=2):
        self.max_iter = max_iter
        self.node_color_dict = {}
        self.node_neighbor_dict = {}

    def setting_init(self, node_list, edge_index):
        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for i in range(edge_index.shape[1]):
            u1, u2 = edge_index[0, i].item(), edge_index[1, i].item()
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def WL_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()  # Using MD5 hash function
                new_color_dict[node] = hashing

            color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]

            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return  
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    def save_coloring(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.node_color_dict, f)

    def visualize_graph(self):
        G = nx.Graph()
        for node, neighbors in self.node_neighbor_dict.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        node_colors = [self.node_color_dict[node] for node in G.nodes()]
        plt.figure(figsize=(8, 8))
        nx.draw(G, node_color=node_colors, with_labels=True, cmap=plt.cm.viridis)
        plt.show()