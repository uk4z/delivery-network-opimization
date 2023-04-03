from scipy.stats import bernoulli

class TreeNode:
    def __init__(self, value, power, distance, parent=None):
        self.parent = parent
        self.children = []
        self.value = value
        self.power = power
        self.distance = distance
        self.broke = False
    
    def broke_with_probability(self, epsilon):
        value = bernoulli(epsilon)
        if  value == 1:
            self.broke = True

class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = {}

    def find_min_power(self, value1, value2, gas_price, broke_routes):
        if value1 == value2:
            return 0, 0
        
        route_distance = 0
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)
        min_power = 0

        node = node1
        while node != lca:
            if node.broke and node!= self.root and broke_routes:
                return None, 0
            
            min_power = max(min_power, node.power)
            route_distance += node.distance
            node = node.parent 
        
        node = node2
        while node != lca:
            if node.broke and node!= self.root and broke_routes:
                return None, 0
            
            min_power = max(min_power, node.power)
            route_distance += node.distance
            node = node.parent

        route_cost = gas_price * route_distance

        return  min_power, route_cost
    
    def find_min_power_path(self, value1, value2, gas_price, broke_routes):
        if value1 == value2:
            return [value1], 0
        
        route_distance = 0
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)

        first_part = []
        node = node1
        while node != lca:
            if node.broke and node!= self.root and broke_routes:
                return None, 0
            
            first_part.append(node.value)
            route_distance += node.distance
            node = node.parent 
        first_part.append(lca.value)

        second_part = []        
        node = node2
        while node != lca:
            if node.broke and node!= self.root and  broke_routes:
                return None, 0
            
            second_part.append(node.value)
            route_distance += node.distance
            node = node.parent
    
        path = first_part + second_part[::-1]

        route_cost = gas_price * route_distance
        return path, route_cost

    def nb_edges_in_route(self, value1, value2):
        if value1 == value2:
            return 0
        
        nb_edges = 0
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)

        node = node1
        while node != lca:
            nb_edges += 1  
            node = node.parent 
        
        node = node2
        while node != lca:
            nb_edges += 1
            node = node.parent

        return nb_edges

def lowest_common_ancestor(node1, node2):
    ancestors = set()

    while node1:
        ancestors.add(node1)
        node1 = node1.parent

    while node2:
        if node2 in ancestors:
            return node2
        node2 = node2.parent

    return None
