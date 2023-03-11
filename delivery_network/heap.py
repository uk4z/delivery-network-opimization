class FibonacciHeap:
    def __init__(self):
        self.min_node = None 
        self.nb_nodes = 0
    
    def __len__(self):
        return self.nb_nodes
    
    class Node:
        def __init__(self, value, key):
            self.value = value 
            self.parent = None 
            self.child = None
            self.left = self.right = self
            self.key = key
        
        def have_neighbors(self):
            if self.right != self:
                return True
            return False
        
        def set_neighbor(self, node):
            if not self.have_neighbors():
                self.right = self.left = node
                node.left = node.right = self

        def add_neighbor(self, node):
            if self.have_neighbors():
                node.left = self
                node.right = self.right
                self.right = node
                node.right.left = node

        def new_neighbor(self, node):
            if self.have_neighbors():
                self.add_neighbor(node)
            else: 
                self.set_neighbor(node)

        def have_child(self):
            if self.child :
                return True
            return False
    
        def set_child(self, node):
            if not self.have_child():
                self.child = node
                node.parent = self

        def add_child(self, node):
            if self.child:
                node.left = self.child
                node.right = self.child.right
                self.child.right = node
                node.right.left = node
                node.parent = self

        def new_child(self, node):
            if self.have_child():
                self.add_child(node)
            else: 
                self.set_child(node)

        def separation_children(self):
            min_child = self.child
            min_child.parent = None
            child = min_child.right
            while  child != self.child :
                if child.key < min_child.key:
                    min_child = child
                child.parent = None  
                child = child.right 
            return min_child
        
        def get_min_neighbor(self):
            min_neighbor = self
            neighbor = self.right
            while neighbor != self:
                if neighbor.key < min_neighbor.key:
                    min_neighbor = neighbor
                neighbor = neighbor.right
            return min_neighbor

        def kill_neighbors(self):
            self.right.left = self.left
            self.left.right = self.right
            self.right = self.left = self 

        def kill_children(self):
            self.child = None 

        def get_depth(self):
            initial_depth = 0

            def depth(node, current_depth):
                if not node.have_child(): 
                    return current_depth
                else:
                    current_depth += 1
                    depths = []
                    child = node.child
                    child_depth = depth(child, current_depth)
                    depths.append(child_depth)
                    neighbor = child.right
                    while neighbor != child:
                        neighbor_depth = depth(neighbor, current_depth)
                        depths.append(neighbor_depth)
                        neighbor = neighbor.right
                    return max(depths)
                
            return depth(self, initial_depth)
        
        def same_depth(self, node):
            return self.get_depth() == node.get_depth() 

        def map(self):
            if self.have_neighbors():
                neighbor = self.right 
                while self.have_neighbors() and neighbor != self:
                    if self.same_depth(neighbor):
                        neighbor.kill_neighbors()
                        self.new_child(neighbor)
                        neighbor = self.right
                    neighbor = neighbor.right

    def is_Empty(self):
        return self.nb_nodes == 0 
    
    def insert_Node(self, value, key):
        new_node = self.Node(value, key)
        if self.is_Empty():
            self.min_node = new_node
        else:
            if key < self.min_node.key:
                new_node.new_child(self.min_node)
                self.min_node = new_node
            elif key > self.min_node.key:
                self.min_node.new_child(new_node)
            else: 
                self.min_node.new_neighbor(new_node)
        self.nb_nodes += 1

    def get_min(self):
        if self.is_Empty():
            return None
        else: 
            return self.min_node 
    
    def union(self, heap):
        self.min_node.new_neighbor(heap.min_node)
        if self.min_node.key > heap.min_node.key:
            self.min_node = heap.min_node
        
    def extract_min(self):
        if not self.get_min():
            return None
        else:
            output = self.min_node.value
            min_node = self.min_node
            if min_node.have_neighbors():
                neighbor = min_node.right 
                min_node.kill_neighbors()
                self.min_node = neighbor.get_min_neighbor()
                if min_node.have_child():
                    heap = FibonacciHeap()
                    min_children = min_node.separation_children()
                    heap.min_node = min_children 
                    min_node.kill_children()
                    self.union(heap)
            return output

    def heap_reorganisation(self):
        min_node = self.min_node
        if min_node.have_neighbors():
            min_node.map()
            neighbor = min_node.right
            while min_node.have_neighbors() and neighbor != min_node:
                neighbor.map()
                neighbor = neighbor.right


