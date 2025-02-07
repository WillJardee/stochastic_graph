from dataclasses import dataclass, field

import numpy as np
import random
from numpy.typing import NDArray
import line_profiler
import networkx as nx
import matplotlib.pyplot as plt

# The number of nodes that are available
@dataclass 
class NodeInfo:
    deg: int
    deg_remain: int = -1
    neighbors: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.deg_remain == -1:
            self.deg_remain = self.deg

@dataclass
class Deg:
    nodes: list[tuple[int, int]]
    sim_matrix: NDArray[np.float64] | np.ma.MaskedArray  # Assumed to triangular
    total_budget: int = 0
    remaining_nodes: int = 0
    
    def __post_init__(self):
        self.node_set = {i: NodeInfo(deg) for (i, deg) in self.nodes}
        self.total_budget = sum([node.deg for node in self.node_set.values()])
        self.remaining_nodes = len(self.nodes)
        
        # Similarity matrix
        self.sim_matrix = self.sim_matrix + self.sim_matrix.T - np.eye(self.sim_matrix.shape[0])
        assert self.sim_matrix.all(), ValueError("Similarity Matrix cannot have off-diagonal zeros")
        if not type(self.sim_matrix) is np.ma.MaskedArray:
            self.sim_matrix_ma = np.ma.masked_array(self.sim_matrix, mask=np.eye(self.sim_matrix.shape[0]))
        vnodes = np.array([[node_idx, node.deg_remain] for node_idx, node in enumerate(self.node_set.values())])
        mask = np.zeros(vnodes.shape)
        self.valid_nodes = np.ma.masked_array(vnodes, mask, dtype=int)

        # Remove nodes with 0 deg
        for node_idx, node in self.node_set.items():
            if node.deg == 0: # then it is already accounted for, so make sure we do not consider it
                self.remove_node(node_idx, builder=True)
                self.valid_nodes[node_idx, :].mask = 1

    # @line_profiler.profile
    def remove_node(self, node_idx, builder=False):
        if self.node_set[node_idx].deg_remain != 0:
            raise ValueError("Tried to remove node with out nodes still available")
        else:
            self.valid_nodes.mask[node_idx, :] = 1
            self.remaining_nodes -= 1
            self.sim_matrix_ma.mask[:,node_idx] = 1
            self.sim_matrix_ma.mask[node_idx,:] = 1

    # @line_profiler.profile
    def decrement_edge(self, node_idx, val:int = 1):
        node = self.node_set[node_idx]
        if node.deg_remain == 0:
            raise ValueError("Tried to decrement node with no value remaining")
        else:
            node.deg_remain -= 1
            self.valid_nodes.data[node_idx, 1] -= 1
            if node.deg_remain == 0:
                self.remove_node(node_idx)

    # @line_profiler.profile
    def add_edge(self, node1: int, node2: int, directed=False):
        self.node_set[node1].neighbors.append(node2)
        self.decrement_edge(node1)
        if not directed:
            self.node_set[node2].neighbors.append(node1)
            self.decrement_edge(node2)

class SocialNetwork:
    def __init__(self, nodes: Deg, *, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.epsilon_rng = random.Random(self.rng.randint(0, 1000))
        self.nodes = nodes

    # @line_profiler.profile
    def weighted_prob(self, a: NDArray, w: NDArray):
        # I have benchmarked these, and have chosen the clear winner between n=10, 100, 1000
        # p1 /= p1.sum()
        # node1 = int(random.choices(valid_nodes, k=1, weights=p1)[0]) # choices is faster on small sets
        # node1 = int(rng.choice(valid_nodes, p=p1))
        # node1 = int(valid_nodes[np.searchsorted(cumsum, rng.uniform(0, cumsum[-1]))])
        cumsum = np.cumsum(w)
        return a[np.searchsorted(cumsum, self.rng.uniform(0, cumsum[-1]))]

    # @line_profiler.profile
    def _hub_first(self, *, 
                  epsilon = 0.5, 
                  kernel = lambda x: x):
        valid_mask = ~self.nodes.valid_nodes.mask[:, 0]  # Boolean mask for valid entries
        valid_weights = self.nodes.valid_nodes.data[valid_mask, 1]
        valid_nodes = self.nodes.valid_nodes.data[valid_mask, 0]
        if self.epsilon_rng.random() > epsilon: # greedy choice
            node1 = valid_weights.argmax()
            node1 = valid_nodes[node1]
        else:
            valid_weights = kernel(valid_weights)
            node1 = int(self.weighted_prob(valid_nodes, valid_weights))
        valid_mask[node1] = 0
        sim_mat = self.nodes.sim_matrix_ma.data[valid_mask, node1]  # Ignore mask
        p2 = kernel(sim_mat)
        neighbors = self.nodes.valid_nodes.data[valid_mask, 0]  # Ignore mask
        node2 = int(self.weighted_prob(neighbors, p2))
        return node1, node2
    
    # @line_profiler.profile
    def _popular_first(self, *, 
                  epsilon = 0.5, 
                  kernel = lambda x: x):
        valid_mask = ~self.nodes.valid_nodes.mask[:, 0]  # Boolean mask for valid entries
        valid_weights = self.nodes.sim_matrix_ma.data[valid_mask, valid_mask].sum(axis=0)
        valid_nodes = self.nodes.valid_nodes.data[valid_mask, 0]
        if self.epsilon_rng.random() > epsilon: # greedy choice
            node1 = valid_weights.argmax()
            node1 = valid_nodes[node1]
        else:
            p1 = kernel(valid_weights)
            node1 = int(self.weighted_prob(valid_nodes, p1))
        valid_mask[node1] = 0
        sim_mat = self.nodes.sim_matrix_ma.data[valid_mask, node1]  # Ignore mask
        p2 = kernel(sim_mat)
        neighbors = self.nodes.valid_nodes.data[valid_mask, 0]  # Ignore mask
        node2 = int(self.weighted_prob(neighbors, p2))
        return node1, node2

    @line_profiler.profile
    def _mixed_selection(self, *, 
                         hub_weight = 0.5,
                         popular_weight = 0.5,
                         epsilon = 0.5, 
                         kernel = lambda x: x):
        r = self.rng.random()
        if r < hub_weight:
            return self._hub_first(epsilon=epsilon, kernel=kernel)
        else:
            return self._popular_first(epsilon=epsilon, kernel=kernel)

    @line_profiler.profile
    def make_graph(self):
    
        output_graph = np.zeros((self.nodes.sim_matrix.shape), dtype=bool) 
    
        while self.nodes.remaining_nodes > 1: # We want to break when there are no more valid nodes
            # node1, node2 = self._hub_first()
            # node1, node2 = self._popular_first()
            node1, node2 = self._mixed_selection()
            deg.add_edge(node1, node2)
            output_graph[node1,node2] = True
            output_graph[node2, node1] = True
        
        return output_graph, deg.valid_nodes[:, 0], deg.valid_nodes[:, 1]

if __name__ == "__main__":
    SEED = 42
    N = 25 
    R = 1
    SHOW = True 

    for i in range(R):
        rng = np.random.default_rng(seed=SEED)
    
        sim_matrix = np.triu(rng.uniform(size=(N, N)))
    
        # sim_matrix = np.array(
        #             [
        #                 [0,   0.1, 0.3, 0.4],     
        #                 [0,   0,   0.2, 0.4],
        #                 [0,   0,   0,   0.5],
        #                 [0,   0,   0,   0  ],
        #             ]
        #         )
    
        rand_degs = rng.integers(0, N+1, size=N)
        if rand_degs.sum() % 2 == 1: rand_degs[rng.integers(N)] += 1
        degs = [(i, d) for i, d in enumerate(rand_degs)]
    
        deg = Deg(nodes=degs, sim_matrix=sim_matrix)
    
        # deg = Deg(nodes=[(0, 3),
        #                  (1, 0),
        #                  (2, 2),
        #                  (3, 1),
        #                 ], 
        #           sim_matrix=sim_matrix)
        
   
        sn = SocialNetwork(nodes=deg, seed=SEED)
        graph, left_over, left_edges = sn.make_graph()
        print(f"Run {i} Success")
   
    if SHOW:
        graph = nx.from_numpy_array(graph)

        # Draw the graph
        plt.figure(figsize=(6,6))
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
        plt.show()

