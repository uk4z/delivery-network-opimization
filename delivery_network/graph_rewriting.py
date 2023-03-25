import sys 
sys.path.append("D:\Coding files\Delivery network\Fibonacci Heap")
from fibonacci_heap import FibonacciHeap
import collections

class GraphEdge:
    def __init__(self, node1=None, node2=None, power=0, distance=1):
        self.node1 = node1
        self.node2 = node2
        self.power = power
        self.distance = distance
    
    def __str__(self):
        source = self.node1.value
        destination = self.node2.value

        return f"{source} --- {destination}"
    
    def find_neighbor_in_edge(self, node):
            if self.node1 != node:
                return self.node1
            
            else: 
                return self.node2

class GraphNode:
    def __init__(self, value, neighbors=None):
        self.value = value
        self.neighbors = neighbors or {}
    
    def is_connected(self, target):
        visited = set()
        stack = [self]

        while stack:
            node = stack.pop()

            if node == target:
                return True
            
            visited.add(node)
                
            for neighbor in node.neighbors.keys():
                if neighbor not in visited:
                    stack.append(neighbor)

        return False

    def set_neighbor(self, node, edge):
        if node not in self.neighbors:
            self.neighbors[node] = [edge]

        else:
            self.neighbors[node].append(edge)
         
           
class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def __str__(self):
        if not self.nodes.keys():
            output = "The graph is empty."

        else:
            output = f"The graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n"
            for edge in self.edges:
                source = edge.node1.value
                destination = edge.node2.value
                output += f"{source} --- {destination}\n"
        
        return output

    def get_node_by_value(self, value):
        return self.nodes[value] 
    
    def get_path(self, source, destination):
        path, distance = self.shortest_path(source, destination)

        if distance !=  -1:
            return path, distance 
        
        else:
            path, distance = self.shortest_path(destination, source)
            new_path = collections.deque()

            for node in path:
                new_path.appendleft(node)

            return new_path, distance

    def shortest_path(self, source, destination): 
        peres = {node : None for node in self.nodes.values()}
        distances = {node : -1 for node in self.nodes.values()}
        distances[source] = 0
        already_processed = {node : False for node in self.nodes.values()}

        if not source.is_connected(destination):
            raise Exception("Nodes are not connected.")
        
        dijkstra(source, peres, distances, already_processed)

        distance = distances[destination]
        path = get_path_from_peres(source, destination, peres)
        return path, distance 
    

def dijkstra(source, peres, distances, already_processed):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()
        already_processed[node] = True

        visit_neighbors(node, heap, peres, distances, already_processed)

def visit_neighbors(node, heap, peres, distances, already_processed):
        for edges in node.neighbors.values():
            for edge in edges:
                neighbor = edge.find_neighbor_in_edge(node)

                if (not already_processed[neighbor]
                    and (distances[neighbor] == -1 
                        or distances[neighbor] >= distances[node] + edge.distance)):
                    peres[neighbor] = node
                    distances[neighbor] = distances[node] + edge.distance

                    if heap.have_wrap(neighbor):
                        heap.decrease_key(neighbor, distances[neighbor])
                        
                    else:
                        heap.insertion(neighbor, distances[neighbor])
            
def get_path_from_peres(source, destination, peres):
    path = collections.deque()
    path.append(destination.value)
    
    node = destination

    while node != source: 
        node = peres[node]

        if node is None:
            return []
        
        path.appendleft(node.value)
    
    return path





