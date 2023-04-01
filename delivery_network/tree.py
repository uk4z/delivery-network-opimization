class TreeNode:
    def __init__(self, value, power, parent=None):
        self.parent = parent
        self.children = []
        self.value = value
        self.power = power


class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = {}

    def find_min_power(self, value1, value2):
        if value1 == value2:
            return -1
        
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)
        min_power = 0

        node = node1
        while node != lca:
            min_power = max(min_power, node.power)
            node = node.parent 
        
        node = node2
        while node != lca:
            min_power = max(min_power, node.power)
            node = node.parent
    
        return  min_power
    
    def find_min_power_path(self, value1, value2):
        if value1 == value2:
            return []
        
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)

        first_part = []
        node = node1
        while node != lca:
            first_part.append(node.value)
            node = node.parent 
        first_part.append(lca.value)

        second_part = []        
        node = node2
        while node != lca:
            second_part.append(node.value)
            node = node.parent
    
        path = first_part + second_part[::-1]

        return path 


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
