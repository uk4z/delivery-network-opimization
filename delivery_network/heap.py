import math

# Creating a class to represent a node in the heap

class FibonacciHeap:
    def __init__(self):
        # Creating min pointer as "mini"
        self.mini = None
        # Declare an integer for number of nodes in the heap
        self.nb_nodes = 0

    class Node:
        def __init__(self, wrap, key):
            self.wrap = wrap
            self.parent = None # Parent pointer
            self.child = None # Child pointer
            self.left = self.right = self # Pointer to the node on the left and on the right
            self.key = key # Value of the node
            self.degree = 0 # Degree of the node
            self.mark = '' # Black or white mark of the node
            self.has_been_visited = False # Flag for assisting in the Find node function
        
        # Linking the heap nodes in parent child relationship
        def Fibonnaci_link(self, parent):
            if self == parent: 
                return
            self.left.right = self.right
            self.right.left = self.left
            self.right = self.left = self
            self.parent = parent
            if (parent.child == None):
                parent.child = self
            self.right = parent.child
            self.left = parent.child.left
            parent.child.left.right = self
            parent.child.left = self
            if self.key < parent.child.key:
                parent.child = self
            parent.degree+=1
        
        # Function to find the given node
        def Find(self, key):
            if (not self) or ((not self.child) and self.right.has_been_visited):
                return False 
            if (self.key == key):
                return True
            
            self.has_been_visited = True 
            if self.child and self.right.has_been_visited:
                return self.child.Find(key)
            elif (not self.right.has_been_visited) and (not self.child):
                return self.right.Find(key)
            elif self.child and (not self.right.has_been_visited):
                return self.child.Find(key) or self.right.Find(key)


        
            
    def is_empty(self):
        return self.nb_nodes == 0 
    
    # Function to insert a node in heap
    def insertion(self, wrap, key):
        new_node = FibonacciHeap.Node(wrap, key)
        new_node.mark = 'W'
        new_node.left = new_node
        new_node.right = new_node
        if (self.mini != None):
            self.mini.left.right = new_node
            new_node.right = self.mini
            new_node.left = self.mini.left
            self.mini.left = new_node
            if (new_node.key < self.mini.key):
                self.mini = new_node
        else:
            self.mini = new_node
        self.nb_nodes+=1

    def have_node(self, wrap_node):
        if self.is_empty():
            return False
        return find(self.mini, wrap_node)
    
    def get_node_by_value(self, wrap):
        if self.is_empty():
            return None 
        return get_node_by_key(self.mini, wrap)

        
    # Consolidating the heap
    def Consolidate(self):
        
        max_degree = int(math.log2(self.nb_nodes)) + 1
        arr = [None] * (max_degree + 1)
        mini = self.mini
        ptr4 = mini
        while True:
            ptr4 = ptr4.right
            degree = mini.degree
            while (arr[degree] != None):
                node = arr[degree]
                if (mini.key > node.key):
                    ptr3 = mini
                    mini = node
                    node = ptr3
                if (node == self.mini):
                    self.mini = mini
                node.Fibonnaci_link(mini)
                if (mini.right == mini):
                    self.mini = mini
                arr[degree] = None
                degree+=1
            arr[degree] = mini
            mini = mini.right
            if (mini == self.mini):
                break

        self.mini = None
        for degree in range(max_degree+1):
            if (arr[degree] != None):
                arr[degree].left = arr[degree]
                arr[degree].right = arr[degree]
                if (self.mini != None) :
                    self.mini.left.right = arr[degree]
                    arr[degree].right = self.mini
                    arr[degree].left = self.mini.left
                    self.mini.left = arr[degree]
                    if (arr[degree].key < self.mini.key):
                        self.mini = arr[degree]
                else:
                    self.mini = arr[degree]
                if self.mini == None:
                    self.mini = arr[degree]
                elif arr[degree].key < self.mini.key:
                    self.mini = arr[degree]
        

    # Function to extract self.minimum node in the heap
    def extract_min(self):
        if self.is_empty():
            return None 
        else:
            mini = self.mini
            output = mini.wrap
            pntr = mini
            if (mini.child != None):

                child = mini.child
                while(True):
                    pntr = child.right
                    self.mini.left.right = child
                    child.right = self.mini
                    child.left = self.mini.left
                    self.mini.left = child
                    if child.key < self.mini.key:
                        self.mini = child
                    child.parent = None
                    child = pntr
                    if (pntr == mini.child):
                        break

            mini.left.right = mini.right
            mini.right.left = mini.left
            self.mini = mini.right
            if mini == mini.right and mini.child == None:
                self.mini = None
            else:
                self.mini = mini.right
                self.Consolidate()
            self.nb_nodes-=1
        return output



    # Cutting a node in the heap to be placed in the root list
    def Cut(self, node, parent):
        if (node == node.right):
            parent.child = None

        node.left.right = node.right
        node.right.left = node.left
        if (node == parent.child):
            parent.child = node.right

        parent.degree -= 1
        node.right = node
        node.left = node
        self.mini.left.right = node
        node.right = self.mini
        node.left = self.mini.left
        self.mini.left = node
        node.parent = None
        node.mark = 'B'

    # Recursive cascade cutting function
    def Cascase_cut(self, node):

        parent = node.parent
        if (parent != None):
            if (node.mark == 'W'):
                node.mark = 'B'
            else:
                self.Cut(node, parent)
                self.Cascase_cut(parent)

    # Function to decrease the value of a node in the heap
    def Decrease_key(self, node_key, new_key):
        if self.is_empty():
            raise Exception("The Heap is Empty")

        if not self.have_node(node_key):
            raise Exception("Node is not in the Heap")

        node = self.get_node_by_value(node_key)
        node.key = new_key

        parent = node.parent
        if (parent != None and node.key < parent.key):
            self.Cut(node, parent)
            self.Cascase_cut(parent)

        if node.key < self.mini.key:
            self.mini = node


    # Deleting a node from the heap
    def Deletion(self, key):

        if (self.mini == None):
            print("The heap is empty")
        else:
            # Decreasing the value of the node to 0
            self.mini.Find(key, 0)

            # Calling Extract_min function to
            # delete self.minimum value node, which is 0
            self.Extract_min()
            print("Key Deleted")


    # Function to display the heap
    def display(self):
        ptr = self.mini
        if (ptr == None):
            print("The Heap is Empty")

        else:
            print("The root nodes of Heap are: ")
            while(True):
                print(ptr.wrap.value,end='')
                ptr = ptr.right
                if (ptr != self.mini):
                    print("-->",end='')
                if not(ptr != self.mini and ptr.right != None):
                    break
            print()
            print(f"The heap has {self.nb_nodes} nodes")


def find(node, wrap):
    if not node:
        return False
    neighbor = node
    while True:
        find_node = find(neighbor.child, wrap)
        if neighbor.wrap == wrap or find_node:
            return True
        neighbor = neighbor.right
        if neighbor == node:
            return False
        
def get_node_by_key(node, wrap):
    if not node:
        return None
    neighbor = node
    while True:
        if neighbor.wrap == wrap:
            return neighbor.wrap
        get_node = get_node_by_key(neighbor.child, wrap)
        if get_node is not None:
            return get_node
        neighbor = neighbor.right
        if neighbor == node:
            return None

