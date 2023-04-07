import numpy as np

class TreeNode:
    def __init__(self, value, power, distance, parent=None):
        self.value = value
        self.power = power
        self.distance = distance
        self.parent = parent
        self.degree = 0
        self.children = []
        self.broke = False
    
    def broke_with_probability(self, epsilon):
        value = np.random.binomial(1, epsilon)
        if value == 1:
            self.broke = True


class Tree:
    def __init__(self, root):
        self.root = root
        self.gas_price = 0
        self.nodes = {}

    def route_characteristics(self, value1, value2, broke=True, path=False, power=False, distance=False, cost=False, number_of_edges=False):
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]

        lca = lowest_common_ancestor(node1, node2)

        first_part, power1_value, distance1_value, route_available1 = self.characteristics_until_lca(node1, lca, broke)
        second_part, power2_value, distance2_value, route_available2 = self.characteristics_until_lca(node2, lca, broke)

        path_nodes = first_part + [lca.value] + second_part[::-1]
        power_value = power1_value + power2_value
        distance_value = distance1_value + distance2_value
        route_cost = distance_value * self.gas_price
        nb_edges = len(path_nodes)-1
        route_available = route_available1 and route_available2
        

        output = {"path" : [path_nodes, path], 
                  "power" : [power_value, power], 
                  "distance" : [distance_value, distance], 
                  "edges" : [nb_edges, number_of_edges],
                  "cost" : [route_cost, cost],
                  "available" : [route_available, broke]}

        return [output[key][0] for key in ["path", "power", "distance", "cost", "edges", "available"] if output[key][1]]

    def characteristics_until_lca(self, node, lca, broke=True):
        path = []
        power = 0
        distance = 0
        route_available = True
        nodes = self.iterate_from_node_to_lca(node, lca)
        for node in nodes:
            if node.broke and broke:
                route_available = False
                            
            if node != lca:
                power = max(power, node.power)
                distance += node.distance
                path.append(node.value)
        
        return path, power, distance, route_available
    
    def iterate_from_node_to_lca(self, start_node, lca):
        node = start_node

        while node != lca:    
            yield node
            node = node.parent

        yield lca


def lowest_common_ancestor(node1, node2):
    while node1.degree > node2.degree:
        node1 = node1.parent

    while node2.degree > node1.degree:
        node2 = node2.parent

    while node1 != node2:
        node1 = node1.parent
        node2 = node2.parent

    return node1

