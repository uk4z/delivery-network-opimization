import math

# Creating a class to represent a node in the heap

class FibonacciHeapNode:
    def __init__(self, wrap, key):
        self.wrap = wrap
        self.key = key
        self.parent = None
        self.child = None 
        self.left = self.right = self
        self.degree = 0 
        self.mark = False 
    
    def suppress_neighbors(self):
        self.left.right = self.right
        self.right.left = self.left 
        self.right = self.left = self
    
    def link_to_the_left(self, node):
        if self.right != self:
            raise Exception("The node has at least one neighbor.")
        
        node.left.right = self 
        self.left = node.left 
        self.right = node
        node.left = self 

    def link_as_child(self, parent):
        if self != parent: 
            self.suppress_neighbors()

            if parent.child is not None:
                child = parent.child
                self.link_to_the_left(child)
            else:
                parent.child = self

            self.parent = parent
            parent.degree += 1
 

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.nb_nodes = 0
    
    def update_min_node(self, node):
        if node.key < self.min_node.key:
            self.min_node = node

    def insertion(self, wrap, key):
        new_node = FibonacciHeapNode(wrap, key)

        if self.min_node is not None:
            new_node.link_to_the_left(self.min_node)
            self.update_min_node(new_node)

        else:
            self.min_node = new_node
            
        self.nb_nodes += 1
    
    def have_wrap(self, wrap):
        if self.min_node is None:
            return False
        
        return dfs_node(self.min_node, wrap) is not None
    
    def _get_node_by_wrap(self, wrap):
        if self.min_node is None:
            return None 
        
        return dfs_node(self.min_node, wrap)

    def _add_child_to_root_list(self):
        if self.min_node.child is not None:
                child = self.min_node.child

                while True:
                    neighbor = child.right

                    child.suppress_neighbors()
                    child.link_to_the_left(self.min_node)
                    child.parent = None

                    child = neighbor

                    if neighbor == self.min_node.child:
                        break

                self.min_node.child = None

    def _extract_only_node(self):
        extracted_node = self.min_node
        self.min_node = None
        self.nb_nodes = 0  

        return extracted_node.wrap

    def extract_min(self):
        if self.min_node is not None:
            self._add_child_to_root_list()

            if self.min_node == self.min_node.right:
                return self._extract_only_node()
            
            extracted_node = self.min_node
            self.min_node = self.min_node.right
            extracted_node.suppress_neighbors()

            self.nb_nodes -= 1
            self.consolidate()
            
            return extracted_node.wrap
    
    def _link_same_degree_nodes(self, root_list):
        node = self.min_node
        while True:
            degree = node.degree

            while root_list[degree]:
                neighbour = root_list[degree]

                if neighbour.key < node.key:
                    neighbour, node = node, neighbour

                if neighbour == self.min_node:
                    self.min_node = node

                neighbour.link_as_child(node)

                if node.right == node:
                    self.min_node = node

                root_list[degree] = None
                degree += 1

            root_list[degree] = node
            node = node.right

            if node == self.min_node:
                break

    def _set_min_node_from_root_list(self, root_list):
        self.min_node = None
        for node in root_list:
            if node:
                if self.min_node is None:
                    self.min_node = node
                else:
                    node.suppress_neighbors()
                    node.link_to_the_left(self.min_node)
                    
                    self.update_min_node(node)

    def consolidate(self):
        MAX_DEGREE = 2 * int(math.log2(self.nb_nodes)) + 1
        root_list = [None] * (MAX_DEGREE + 1)

        self._link_same_degree_nodes(root_list)
        self._set_min_node_from_root_list(root_list)

        

    def _cut(self, node, parent):
        if node == parent.child:
            parent.child = node.right if (node.right != node) else None

        node.mark = False 
        node.parent = None
        node.suppress_neighbors()
        node.link_to_the_left(self.min_node)
        
        parent.degree -= 1

    def _cascade_cut(self, node):
        parent = node.parent

        if parent is not None:
            if not node.mark :
                node.mark = True

            else:
                self._cut(node, parent)
                self._cascade_cut(parent)

    def decrease_key(self, node_wrap, new_key):
        node = self._get_node_by_wrap(node_wrap)
        node.key = new_key

        parent = node.parent
        
        if (parent is not None) and node.key < parent.key:
            self._cut(node, parent)
            self._cascade_cut(parent)

        self.update_min_node(node)

def dfs_node(node, wrap):
    stack = [node]
    visited = set()

    while stack:
        current_node = stack.pop()

        if current_node not in visited:
            visited.add(current_node)

            if current_node.wrap == wrap:
                return current_node

            if current_node.child:
                stack.append(current_node.child)

            if current_node.right != current_node:
                stack.append(current_node.right)

    return None
