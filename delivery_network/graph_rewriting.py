def graph_from_file(filename):
    graph_nodes, graph_edges = open_file(filename)
    graph = Graph()
    set_nodes_to_graph(graph_nodes, graph)
    set_edges_to_graph(graph_edges, graph)
    print(show_graph(graph))
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
    edges = []
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

    return edges

def add_edge_to_Graph(edge, graph):
    graph.edges.append(edge)
    node1 = edge.node1
    node2 = edge.node2
    node1.edges.append(edge)
    node2.edges.append(edge)

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
        
        def set_neighbor(self, node):
            if node not in self.neighbors:
                self.neighbors[node.value] = node 

        def shortest_route_with_power(self, destination): 
            self.peres = {}
            self.distances = {self : 0}
            self.already_processed = {self : True}
            graph.dijkstra(self)
            end_to_start_route = [destination.value]
            distance = self.distances[destination]
            while destination != self: 
                destination = self.peres[destination]
                end_to_start_route.append(destination.value)
            start_to_end_route = end_to_start_route[::-1]
            return [start_to_end_route, distance]  
        
        def visit_neighbors(root, node, reach_nodes):
            for edge in node.edges:
                neighbor = edge.find_neighbor_in_edge(node)
                if (neighbor not in root.already_processed 
                    and (neighbor not in root.distances 
                        or root.distances[neighbor] > root.distances[node] + edge.distance)):
                    root.peres[neighbor] = node
                    root.distances[neighbor] = root.distances[node] + edge.distance
                    heap_node = reach_nodes.get_node_in_heap(neighbor.value)
                    if not reach_nodes.have_node(heap_node):
                        reach_nodes.insert_Node(neighbor.value, root.distances[neighbor])
                    else:
                        reach_nodes.decrease_key(heap_node, root.distances[neighbor])

        def connected_components(self):
            visit = set() 
            def helper(node):
                if node.value in visit:
                    return
                else:
                    visit.add(node.value)
                    for neighbor in node.neighbors.values():
                        helper(neighbor)
            helper(self)
            return visit

    def connected_components_set(self):
        output = set()
        for node in self.nodes.values():
            connected_nodes = node.connected_components()
            if connected_nodes not in output:
                output.add(frozenset(connected_nodes))
        return output

    def get_node(self,value):
        return self.nodes[value] 
            
    def dijkstra(self, root):
        reach_nodes = FibonacciHeap()
        reach_nodes.insert_Node(root.value, 0)
        while not reach_nodes.is_Empty():
            node_value = reach_nodes.extract_min()
            node = self.get_node(node_value)
            root.already_processed[node] = True
            root.visit_neighbors(node, reach_nodes)

    def nb_nodes(self):
        return len(self.nodes)
    
    def nb_edges(self):
        return len(self.edges)
    
import sys 
sys.path.append("D:\Coding files\Delivery network\Fibonacci Heap")
from heap import FibonacciHeap

network_filename = "input/network.00.in"


graph = graph_from_file(network_filename)

print(graph.nodes[1].shortest_route_with_power(graph.nodes[3]))
