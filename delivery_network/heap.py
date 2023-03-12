class FibonacciHeap:
    def __init__(self):
        self.min_node = None 
    
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
        
        def get_min_child(self):
            if not self.have_child():
                return None 
            min_child = self.child
            child = min_child.right
            while  child != self.child :
                if child.key < min_child.key:
                    min_child = child
                child = child.right
            return min_child
        
        def kill_children(self):
            child = self.child.right 
            while  child != self.child :
                child.parent = None  
                child = child.right 
            self.child.parent = None
            self.child = None 
        
        def get_min_neighbor(self):
            if not self.have_neighbors():
                return None
            neighbor = self.right
            min_neighbor = neighbor
            while neighbor != self:
                if neighbor.key < min_neighbor.key:
                    min_neighbor = neighbor
                neighbor = neighbor.right
            return min_neighbor

        def kill_neighbors(self):
            self.right.left = self.left
            self.left.right = self.right
            self.right = self.left = self 

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

        def change_child_from_children(self):
            if self.have_child():
                self.child = self.child.right
        
        def have_this_neighbor(self, node):
            if self.have_neighbors():
                neighbor = self.right
                while neighbor != self:
                    if neighbor == node:
                        return True 
            return False 
        
        def neighbors(self):
            neighbors = [self]
            if self.have_neighbors():
                neighbor = self.right
                while neighbor != self:
                   neighbors.append(neighbor)
                   neighbor = neighbor.right
            return neighbors

        def _find_node(self, root_list, node):
            if self is None or not root_list:
                return False
            if node in root_list:
                return True
            else :
                for root in root_list:
                    if root.have_child():
                        child_list = root.child.neighbors()
                    else:
                        child_list = [] 
                    if root._find_node(child_list, node):
                        return True
                return False 
            
        def _get_node_by_value(self, root_list, node_value):
            if self is None or not root_list:
                return None
            for root in root_list:
                if root.value == node_value:
                    return root
            else :
                for root in root_list:
                    if root.have_child():
                        child_list = root.child.neighbors()
                    else:
                        child_list = [] 
                    node = root._get_node_by_value(child_list, node_value)
                    if node is not None:
                        return node
        
    def is_Empty(self):
        return self.min_node is None 
    
    def have_node(self, node):
        if not self.min_node:
            return False
        min_node = self.min_node
        neighbors = min_node.neighbors()
        return min_node._find_node(neighbors, node)
    
    def get_node_in_heap(self, node_value):
        if not self.min_node:
            return None
        min_node = self.min_node
        neighbors = min_node.neighbors()
        return min_node._get_node_by_value(neighbors, node_value)
    
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

    def get_min(self):
        if self.is_Empty():
            return None
        else: 
            return self.min_node 
    
    def union(self, heap):
        if not self.min_node:
            self.min_node = heap.min_node

        self.min_node.new_neighbor(heap.min_node)
        if self.min_node.key > heap.min_node.key:
            self.min_node = heap.min_node
        else:
            heap.min_node = self.min_node

    def heap_reorganisation(self):
        if not self.is_Empty():
            min_node = self.min_node
            if min_node.have_neighbors():
                min_node.map()
                neighbor = min_node.right
                while min_node.have_neighbors() and neighbor != min_node:
                    neighbor.map()
                    neighbor = neighbor.right
                
    def cut(self, node, parent):
        if node == parent.child:
            if node.have_neighbors():
                parent.change_child_from_children()
                node.kill_neighbors()
            else :
                parent.child = None  
            node.parent = None      
        self.min_node.new_neighbor(node)

    def decrease_key(self, node, new_key):
        node.key = new_key
        parent = node.parent
        if parent and node.key < parent.key:
            self.cut(node, parent)
        if node.key < self.min_node.key:
            self.min_node = node

    def extract_min(self):
        if not self.get_min():
            return None
        else:
            output = self.min_node.value
            min_node = self.min_node
            self.min_node = min_node.get_min_neighbor()
            min_node.kill_neighbors()
            
            if min_node.have_child():
                heap = FibonacciHeap()
                min_child = min_node.get_min_child()
                min_node.kill_children()
                heap.min_node = min_child
                self.union(heap)
            self.heap_reorganisation()

            return output
