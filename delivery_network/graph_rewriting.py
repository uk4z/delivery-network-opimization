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
    
    def find_neighbour_in_edge(self, node):
            if self.node1 != node:
                return self.node1
            
            else: 
                return self.node2


class GraphNode:
    def __init__(self, value, neighbours=None):
        self.value = value
        self.neighbours = neighbours or {}

    def is_connected(self, target):
        visited = set()
        stack = [self]

        while stack:
            node = stack.pop()

            if node == target:
                return True
            
            visited.add(node)
                
            for neighbour in node.neighbours.keys():
                if neighbour not in visited:
                    stack.append(neighbour)

        return False

           
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

    def _get_node_by_value(self, value):
        return self.nodes[value] 
    
    def get_path(self, source_value, destination_value):
        source = self._get_node_by_value(source_value)
        destination = self._get_node_by_value(destination_value)

        if not source.is_connected(destination):
            return None
        
        path, distance = self._shortest_path(source, destination)

        if distance !=  -1:
            return path, distance 
        
        else:
            path, distance = self._shortest_path(destination, source)
            new_path = collections.deque()

            for node in path:
                new_path.appendleft(node)

            return new_path, distance

    def _shortest_path(self, source, destination): 
        peres = {node : None for node in self.nodes.values()}
        distances = {node : -1 for node in self.nodes.values()}
        distances[source] = 0
        already_processed = {node : False for node in self.nodes.values()}
        
        dijkstra(source, peres, distances, already_processed)

        distance = distances[destination]
        path = get_path_from_peres(source, destination, peres)

        return path, distance 
    
    def connected_components(self, node_value):
        node = self._get_node_by_value(node_value)

        visited = set()
        stack = [node]

        while stack:
            node = stack.pop()
            
            visited.add(node.value)
                
            for neighbour in node.neighbours.keys():
                if neighbour.value not in visited:
                    stack.append(neighbour)

        return visited

    def connected_components_set(self):
        result = set()

        for node in self.nodes.values():
            if len(result) == len(self.nodes):
                return result 
            
            if node.value not in result:
                connect = self.connected_components(node.value)
                result.add(frozenset(connect))

        return result
             
             
def dijkstra(source, peres, distances, already_processed):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()
        already_processed[node] = True

        visit_neighbours(node, heap, peres, distances, already_processed)

def visit_neighbours(node, heap, peres, distances, already_processed):
        for edges in node.neighbours.values():
            for edge in edges:
                neighbour = edge.find_neighbour_in_edge(node)

                if (not already_processed[neighbour]
                    and (distances[neighbour] == -1 
                        or distances[neighbour] >= distances[node] + edge.distance)):
                    peres[neighbour] = node
                    distances[neighbour] = distances[node] + edge.distance

                    if heap.have_wrap(neighbour):
                        heap.decrease_key(neighbour, distances[neighbour])
                        
                    else:
                        heap.insertion(neighbour, distances[neighbour])
            
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


def graph_from_file(filename):
    nb_nodes, edges_of_graph = open_network_file(filename)
    graph = Graph()

    set_nodes_to_graph(nb_nodes, graph)
    set_edges_to_graph(edges_of_graph, graph)

    return graph

def open_network_file(network):
    with open(network, 'r') as file:
        input = file.read()
        lines = input.split("\n")
        first_line = lines[0]

        nb_nodes = get_data_from_line(first_line)[0]
        edges_of_graph = lines[1:]

        return nb_nodes, edges_of_graph

def set_nodes_to_graph(nb_nodes, graph):
    for value in range(1, nb_nodes + 1):
        graph.nodes[value] = GraphNode(value) 

def get_data_from_line(line):
    raw_data = line.split(' ')
    
    return [int(data) for data in raw_data]

def set_edges_to_graph(graph_edges, graph):
    for line in graph_edges:
        edge = get_data_from_line(line)
        nb_info = len(edge)

        if nb_info == 4:
            node1_value, node2_value, power, distance = edge

        if nb_info == 3:
            node1_value, node2_value, power = edge
            distance = 1

        node1 = graph._get_node_by_value(node1_value)
        node2 = graph._get_node_by_value(node2_value)
        
        new_edge = GraphEdge(node1, node2, power, distance)
        add_edge_to_Graph(graph, new_edge)

def add_edge_to_Graph(graph, edge):
    graph.edges.append(edge)
    
    set_edge_to_nodes(edge)

def set_edge_to_nodes(edge):
        node1 = edge.node1
        node2 = edge.node2

        if node1 not in node2.neighbours:
            node2.neighbours[node1] = [edge]
            node1.neighbours[node2] = [edge]

        else:
            node2.neighbours[node1].append(edge)
            node1.neighbours[node2].append(edge)



