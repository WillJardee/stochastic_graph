from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_array
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
    remaining_budget: int = 0
    remaining_nodes: int = 0
    # valid_nodes: np.ma.MaskedArray = np.ma.masked_array()
    # need to complete this type hinting
    valid_nodes : list[int] = field(default_factory=list) 
    valid_nodes_deg : list[int] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_set = {i: NodeInfo(deg) for (i, deg) in self.nodes}
        self.total_budget = sum([node.deg for node in self.node_set.values()])
        self.remaining_nodes = len(self.nodes)
        self.remaining_budget = self.total_budget
        # self.valid_nodes = np.ones(self.remaining_nodes) 
        # self.valid_nodes = np.ma.masked_array([[idx, node.deg] for idx, node in self.node_set.items() if ((node.deg_remain != None) and (node.deg_remain > 0))]) # using a maksed array means that we are running in O(n) at every step to get valid nodes 
        # self.valid_nodes = [idx for idx, node in self.node_set.items() if ((node.deg_remain != None) and (node.deg_remain > 0))] # this could also be done with bool(node.deg_remain), but that is less clear  
        # self.valid_nodes = [node.deg_remain for _, node in self.node_set.items() if ((node.deg_remain != None) and (node.deg_remain > 0))] # this could also be done with bool(node.deg_remain), but that is less clear  
        
        # Similarity matrix
        self.sim_matrix = self.sim_matrix + self.sim_matrix.T - np.eye(self.sim_matrix.shape[0])
        assert self.sim_matrix.all(), ValueError("Similarity Matrix cannot have off-diagonal zeros")
        if not type(self.sim_matrix) is np.ma.MaskedArray:
            self.sim_matrix_ma = np.ma.masked_array(self.sim_matrix, mask=np.eye(self.sim_matrix.shape[0]))
        
        # Remove nodes with 0 deg
        for node_idx, node in self.node_set.items():
            if node.deg == 0: # then it is already accounted for, so make sure we do not consider it
                self.remove_node(node_idx, builder=True)
            else: 
                self.valid_nodes.append(node_idx)
                self.valid_nodes_deg.append(node.deg)

    def remove_node(self, node_idx, builder=False):
        if self.node_set[node_idx].deg_remain != 0:
            raise ValueError("Tried to remove node with out nodes still available")
        else:
            if not builder:
                node_loc = self.valid_nodes.index(node_idx)
                self.valid_nodes.pop(node_loc)
                self.valid_nodes_deg.pop(node_loc) # inefficient
            self.remaining_nodes -= 1
            self.sim_matrix_ma.mask[:,node_idx] = 1
            self.sim_matrix_ma.mask[node_idx,:] = 1

    def decrement_edge(self, node_idx, val:int = 1):
        if self.node_set[node_idx].deg_remain == 0:
            raise ValueError("Tried to decrement node with no value remaining")
        else:
            self.node_set[node_idx].deg_remain -= 1
            self.remaining_budget -= 1
            node_loc = self.valid_nodes.index(node_idx)
            self.valid_nodes_deg[node_loc] = self.node_set[node_idx].deg_remain # inefficient
            if self.node_set[node_idx].deg_remain == 0:
                self.remove_node(node_idx)

    def add_edge(self, node1: int, node2: int, directed=False):
        self.node_set[node1].neighbors.append(node2)
        self.decrement_edge(node1)
        if not directed:
            self.node_set[node2].neighbors.append(node1)
            self.decrement_edge(node2)

@line_profiler.profile
def hub_first(nodes, rng, epsilon_rng, *, 
              epsilon = 0.1, 
              kernel = lambda x: x):
    if rng.random() > epsilon: # greedy choice
        node_weights = np.array(nodes.valid_nodes_deg)
        node1 = nodes.valid_nodes[node_weights.argmax()]
    else:
        # p1 = [i/deg.remaining_budget for i in nodes.valid_nodes_deg]
        node_weights = kernel(np.array(nodes.valid_nodes_deg))
        p1 = node_weights/node_weights.sum()
        node1 = int(rng.choice(nodes.valid_nodes, 1, replace=False, p=p1)[0])
    sim_mat = deg.sim_matrix_ma[:,node1]
    sim_mat = sim_mat.compressed()
    p2 = kernel(sim_mat)
    # p2 = kernel(deg.sim_matrix_ma[node1,:].compressed())
    p2 /= sum(p2)
    neighbors = [i for i in nodes.valid_nodes if i != node1] #TODO: make this better
    node2 = int(rng.choice(neighbors, 1, p=p2/sum(p2))[0])
    return node1, node2

# @line_profiler.profile
def popular_first(nodes, rng, epsilon_rng, *, 
                  epsilon: float = 0.1, 
                  kernel = lambda x: x):
    assert (epsilon >= 0) and (epsilon <= 1)
    node_weights = kernel(nodes.sim_matrix_ma.sum(axis=0).compressed())
    if rng.random() > epsilon: # greedy choice
        node1 = nodes.valid_nodes[node_weights.argmax()]
    else:
        p1 = node_weights/sum(node_weights) 
        node1 = int(rng.choice(nodes.valid_nodes, 1, replace=False, p=p1)[0])
    
    # same selection rule for node2 as hub_first
    node_weights = kernel(deg.sim_matrix_ma[node1,:].compressed())
    p2 = node_weights/sum(node_weights)
    neighbors = [i for i in nodes.valid_nodes if i != node1] #TODO: make this better
    node2 = int(rng.choice(neighbors, 1, p=p2/sum(p2))[0])
    return node1, node2

# @line_profiler.profile
def mixed_first(nodes, rng, epsilon_rng, *, 
                hub_weight, 
                popular_weight, 
                epsilon: float = 0.1, 
                kernel = lambda x: x):
    prob = np.array([hub_weight, popular_weight])/np.array([hub_weight, popular_weight]).sum()
    rule_choice = rng.choice([hub_first, popular_first], size=1, p=prob)[0]
    return rule_choice(nodes, rng, epsilon_rng, epsilon=epsilon, kernel=kernel)

@line_profiler.profile
def make_graph(nodes: Deg, selection_rule, *, seed=None):
    rng = np.random.default_rng(seed=seed)
    epsilon_rng = np.random.default_rng(seed=rng.integers(1000)) # allows us to track epsilon selection ind

    output_graph = lil_array((nodes.sim_matrix.shape), dtype=bool) 

    while nodes.remaining_nodes > 1: # We want to break when there are no more valid nodes
        node1, node2 = selection_rule(nodes, rng, epsilon_rng)
        deg.add_edge(node1, node2)
        output_graph[node1,node2] = True
        output_graph[node2, node1] = True
    
    return output_graph, deg.valid_nodes, deg.valid_nodes_deg

if __name__ == "__main__":
    SEED = 42
    N = 1000
    SHOW = False

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
    
    # epsilon = 1 turns of epsilon-greedy
    # epsilon = 0 turns on greedy
    # selection_rule = lambda nodes, rng, epsilon_rng: mixed_first(nodes, rng, epsilon_rng, hub_weight=0.5, popular_weight=0.5, epsilon=0.1, kernel=lambda x: np.exp(x))
    selection_rule = lambda nodes, rng, epsilon_rng: hub_first(nodes, rng, epsilon_rng, epsilon=0)

    # profiler = LineProfiler()
    # profiler.add_function(make_graph)  # Profile only this function
    # profiler.enable()
    graph, left_over, left_edges = make_graph(deg, selection_rule, seed=SEED)
    # profiler.disable()
    # profiler.print_stats()
    # print(graph)
    # print(f"Leftover node: {left_over}")
    # print(f"Leftover edge: {left_edges}")
    
    if SHOW:
        csr_matrix = graph.tocsr()

        # Create NetworkX graph from sparse matrix
        graph = nx.from_scipy_sparse_array(csr_matrix)

        # Draw the graph
        plt.figure(figsize=(6,6))
        nx.draw(graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.show()

