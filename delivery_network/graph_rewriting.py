def graph_from_file(filename):
    graph_nodes, graph_edges = open_file(filename)
    graph = Graph()
    set_nodes_to_graph(graph_nodes, graph)
    set_edges_to_graph(graph_edges, graph)
    return graph

def open_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")
        graph_nodes = lines[0]
        graph_edges = lines[1:]
        return graph_nodes, graph_edges

def set_nodes_to_graph(graph_info, graph):
    nb_nodes = get_data_from_line(graph_info)[0]
    for value in range(1, nb_nodes + 1):
        graph.nodes[value] = Graph.Node(value, graph) 

def get_data_from_line(line):
    raw_data = line.split(' ')
    processed_data = [int(data) for data in raw_data]
    return processed_data

def set_edges_to_graph(graph_edges, graph):
    for line in graph_edges:
        edge = get_data_from_line(line)
        nb_info = len(edge)
        if nb_info == 4:
            node1_value, node2_value, power, distance = edge
        if nb_info == 3:
            node1_value, node2_value, power = edge
            distance = 1
        node1 = graph.get_node(node1_value)
        node2 = graph.get_node(node2_value)
        new_edge = Graph.Edge(graph, node1, node2, power, distance)
        add_edge_to_Graph(new_edge, graph)


def add_edge_to_Graph(edge, graph):
    graph.edges.append(edge)
    node1 = edge.node1
    node2 = edge.node2
    node1.edges.append(edge)
    node1.neighbors[node2.value] = node2
    node2.edges.append(edge)
    node2.neighbors[node1.value] = node1

def show_graph(graph):
    if graph.nb_nodes() == 0:
        output = "The graph is empty."
    else:
        output = f"The graph has {graph.nb_nodes()} nodes and {graph.nb_edges()} edges.\n"
        for edge in graph.edges:
            source = edge.node1.value
            destination = edge.node2.value
            output += f"{source} --- {destination}\n"
    
    return output

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    class Edge:
        def __init__(self, graph, node1=None, node2=None, power=0, distance=1):
            self.node1 = node1
            self.node2 = node2
            self.power = power
            self.distance = distance
            self.graph = graph
        
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
            self.neighbors = neighbors or {}
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
                for neighbor in node.neighbors.values():
                    if neighbor not in visited:
                        stack.append(neighbor)
            return False

        def set_neighbor(self, node):
            if node not in self.neighbors:
                self.neighbors[node.value] = node 
        
        

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
            while destination != self: 
                destination = peres[destination]
                end_to_start_route.append(destination.value)
            start_to_end_route = end_to_start_route[::-1]
            return start_to_end_route, distance 


    def get_node(self,value):
        return self.nodes[value] 
        
    def nb_nodes(self):
        return len(self.nodes)
    
    def nb_edges(self):
        return len(self.edges)
    
def dijkstra(root, peres, distances, already_processed):
    heap = FibonacciHeap()
    heap.insertion(root, 0)
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
                if not heap.have_node(neighbor):
                    heap.insertion(neighbor, distances[neighbor])
                else:
                    heap.Decrease_key(neighbor, distances[neighbor])
            

import sys 
sys.path.append("D:\Coding files\Delivery network\Fibonacci Heap")
from Fibonacci_Tree import *

network_filename = "input/network.12.in"

graph = graph_from_file(network_filename)
#print(graph.nodes[1].shortest_route(graph.nodes[9]))
for i in range(1,21):
    for j in range(1,21):
        print(i, j)
        print(graph.nodes[i].shortest_route(graph.nodes[j]))
