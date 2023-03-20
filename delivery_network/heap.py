import math

# Creating a class to represent a node in the heap

class FibonacciHeapNode:
    def __init__(self, wrap, key):
        self.wrap = wrap
        self.parent = None # Parent pointer
        self.child = None # Child pointer
        self.left = self.right = self # Pointer to the node on the left and on the right
        self.key = key # Value of the node
        self.degree = 0 # Degree of the node
        self.mark = False # Boolean mark of the node
    
    # Linking the heap nodes in parent child relationship

    def suppress_neighbors(self):
        self.left.right = self.right
        self.right.left = self.left 
        self.right = self.left = self
    
    def add_as_neighbor(self, node):
        node.left.right = self 
        self.left = node.left 
        self.right = node
        node.left = self 

    def link_as_child(self, parent):
        if self == parent:
            return 
        self.suppress_neighbors()

        if not parent.child:
            parent.child = self
        else:
            child = parent.child
            self.add_as_neighbor(child)

        self.parent = parent
        parent.degree += 1
 

class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.nb_nodes = 0
        
    def is_empty(self):
        return self.nb_nodes == 0 
    
    def insertion(self, wrap, key):
        new_node = FibonacciHeapNode(wrap, key)

        if not self.min_node:
            self.min_node = new_node
        else:
            new_node.add_as_neighbor(self.min_node)
            if (new_node.key < self.min_node.key):
                self.min_node = new_node

        self.nb_nodes += 1

    def get_min_wrap(self):
        if not self.min_node:
            return None
        return self.min_node.wrap
    
    def have_node(self, wrap_node):
        if not self.min_node:
            return False
        return get_node_by_wrap(self.min_node, wrap_node) is not None
    
    def get_node(self, wrap):
        if not self.min_node:
            return None 
        return get_node_by_wrap(self.min_node, wrap)

        
    # Consolidating the heap
    def consolidate(self):
        max_degree = 2 * int(math.log2(self.nb_nodes)) + 1
        arr = [None] * (max_degree + 1)
        node = self.min_node
        while True:
            degree = node.degree
            while arr[degree]:
                neighbour = arr[degree]
                if neighbour.key < node.key:
                    neighbour, node = node, neighbour
                if neighbour == self.min_node:
                    self.min_node = node
                neighbour.link_as_child(node)
                if node.right == node:
                    self.min_node = node
                arr[degree] = None
                degree += 1

            arr[degree] = node
            node = node.right
            if node == self.min_node:
                break

        self.min_node = None
        for node in arr:
            if node:
                if not self.min_node:
                    self.min_node = node
                else:
                    node.suppress_neighbors()
                    node.add_as_neighbor(self.min_node)
                    
                    if node.key < self.min_node.key:
                        self.min_node = node
             

    # Function to extract self.min_nodemum node in the heap
    def extract_min(self):
        if not self.min_node:
            return None 

        min_node = self.min_node
        output = self.get_min_wrap()

        if min_node.child:
            child = min_node.child
            while True:
                neighbor = child.right
                child.add_as_neighbor(min_node)
                child.parent = None
                child = neighbor

                if neighbor == min_node.child:
                    break
            min_node.child = None

        if min_node == min_node.right:
            self.min_node = None
            self.nb_nodes = 0   
            return output
        self.min_node = self.min_node.right

        min_node.suppress_neighbors()
        self.nb_nodes -= 1
        self.consolidate()
        

        return output

    # cutting a node in the heap to be placed in the root list
    def cut(self, node, parent):
        if node == parent.child:
            parent.child = node.right if node.right != node else None

        node.suppress_neighbors()
        parent.degree -= 1
        node.parent = None
        node.add_as_neighbor(self.min_node)
        node.mark = False 

    # Recursive cascade cutting function
    def cascade_cut(self, node):
        parent = node.parent
        if not parent:
            return 
        if not node.mark :
            node.mark = True
        else:
            self.cut(node, parent)
            self.cascade_cut(parent)

    # Function to decrease the value of a node in the heap
    def Decrease_key(self, node_wrap, new_key):
        if not self.min_node:
            raise Exception("The Heap is Empty")

        if not self.have_node(node_wrap):
            raise Exception("Node is not in the Heap")

        node = self.get_node(node_wrap)
        node.key = new_key

        parent = node.parent
        
        if parent and node.key < parent.key:
            self.cut(node, parent)
            self.cascade_cut(parent)

        if node.key < self.min_node.key:
            self.min_node = node

def get_node_by_wrap(node, wrap):
    stack = [node]
    visited = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current.wrap == wrap:
            return current

        if current.child:
            stack.append(current.child)

        if current.right and current.right != node:
            stack.append(current.right)

    return None
