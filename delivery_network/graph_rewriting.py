import sys 
sys.path.append("D:\Coding files\Delivery network\Fibonacci Heap")
from fibonacci_heap import FibonacciHeap

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    class Edge:
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
                
    class Node:
        def __init__(self, value, graph, neighbors=None, edges=None):
            self.value = value
            self.neighbors = neighbors or []
            self.edges = edges or []
            self.graph = graph
        
        def is_connected(self, target):
            visited = set()
            stack = [self]

            while stack:
                node = stack.pop()

                if node == target:
                    return True
                
                visited.add(node)

                for neighbor in node.neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)

            return False

        def set_neighbor(self, node):
            if node not in self.neighbors:
                self.neighbors.append(node) 
        
        def shortest_route(self, destination): 
            peres = {node : None for node in self.graph.nodes.values()}
            distances = {node : -1 for node in self.graph.nodes.values()}
            distances[self] = 0
            already_processed = {node : False for node in self.graph.nodes.values()}
            if not self.is_connected(destination):
                raise Exception("Nodes are not connected.")
            dijkstra(self, peres, distances, already_processed)
            end_to_start_route = [destination.value]
            distance = distances[destination]
            node = destination
            while node != self: 
                node = peres[node]
                if not node:
                    return [], -1
                end_to_start_route.append(node.value)
            start_to_end_route = end_to_start_route[::-1]
            return start_to_end_route, distance 

    def get_node(self,value):
        return self.nodes[value] 
    
    def shortest_path(self, source, destination):
        output = source.shortest_route(destination)
        if output[1] == -1:
            path = destination.shortest_route(source)
            output = (path[0][::-1], path[1]) 
        return output


def dijkstra(source, peres, distances, already_processed):
    heap = FibonacciHeap()
    heap.insertion(source, 0)
    while heap.min_node:
        node = heap.extract_min()
        already_processed[node] = True
        visit_neighbors(node, heap, peres, distances, already_processed)

def visit_neighbors(node, heap, peres, distances, already_processed):
        for edge in node.edges:
            neighbor = edge.find_neighbor_in_edge(node)
            if (not already_processed[neighbor]
                and (distances[neighbor] == -1 
                    or distances[neighbor] > distances[node] + edge.distance)):
                peres[neighbor] = node
                distances[neighbor] = distances[node] + edge.distance
                if not heap.have_wrap(neighbor):
                    heap.insertion(neighbor, distances[neighbor])
                else:
                    heap.decrease_key(neighbor, distances[neighbor])
            




