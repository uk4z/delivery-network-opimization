from fibonacci_heap import FibonacciHeap
from tree import *

import collections

import time

import networkx as nx
import matplotlib.pyplot as plt


class GraphEdge:
    def __init__(self, node1, node2, power=0, distance=1):
        self.node1 = node1
        self.node2 = node2
        self.power = power
        self.distance = distance
    
    def __str__(self):
        source = self.node1.value
        destination = self.node2.value

        return f"{source} --- {destination}"
    

class GraphNode:
    def __init__(self, value, neighbours=None):
        self.value = value
        self.neighbours = neighbours or {}

    def sort_edges_in_neighbours(self):
        for neighbour, edges in self.neighbours.items():
            sorted_edges = sorted(edges, key=lambda edge: edge.power, reverse=True)
            self.neighbours[neighbour] = sorted_edges

    def is_connected_with_power(self, target, truck_power=float("inf")):
        visited = set()
        stack = [self]

        while stack:
            node = stack.pop()
            visited.add(node)

            if node == target:
                return True
                            
            for neighbour in node.neighbours.keys():
                if neighbour not in visited and node.can_reach_neighbour(neighbour, truck_power):
                    stack.append(neighbour)

        return False

    def can_reach_neighbour(self, neighbour, truck_power):
        if neighbour not in self.neighbours.keys():
            return False
        
        for edge in self.neighbours[neighbour]:
            if truck_power >= edge.power:
                return True
        
        return False


class GraphRoute:
    def __init__(self, source, destination, utility):
        self.source = source
        self.destination = destination
        self.utility = utility
        self.expected_utility = 0 
        self.cost = 0
        self.power = 0
        self.available = True
        


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.routes = []
        self.MST = None
        self.station = None
        self.gas_price = 0
        self.broke_probability = 0

    def __str__(self):
        if not self.nodes.keys():
            output = "The graph is empty."

        else:
            output = f"The graph has {len(self.nodes)} nodes and {len(self.edges)} edges.\n"
            for edge in self.edges:
                source = edge.node1.value
                destination = edge.node2.value
                output += f"{source} --- {destination} with power {edge.power}\n"
        
        return output

    def connected_components(self, node_value):
        node = self.nodes[node_value]

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
            if node.value not in result:
                connect = self.connected_components(node.value)
                result.add(frozenset(connect))

        return result
    
    def sort_edges(self):
        self.edges.sort(key=lambda edge: edge.power)

        for node in self.nodes.values():
            node.sort_edges_in_neighbours()

    def set_route_characteristics(self, route, broke=True): 
        source = route.source.value
        destination = route.destination.value
        epsilon = self.broke_probability

        power, route_cost, nb_edges  = self.MST.route_characteristics(source, destination, broke, 
                                                                      power=True, 
                                                                      cost=True, 
                                                                      number_of_edges=True)
        
        success_proba = (1-epsilon)**nb_edges
        failure_proba = 1-success_proba

        route.expected_utility = route.utility*(success_proba - failure_proba)

        if nb_edges == -1:
            route.available = False
            
        route.power = power       
        route.cost = route_cost    
        
    def get_path_given_power(self, source, destination, truck_power=float("inf")):
        parents = {node : None for node in self.nodes.values()}
        distances = {node: 0 if node == source else -1 for node in self.nodes.values()}
        
        if not source.is_connected_with_power(destination, truck_power):
            return None
        
        dijkstra_with_distance(source, parents, distances, truck_power)

        distance = distances[destination]
        path = get_path_from_parents(source, destination, parents)

        return path, distance 
    
    def get_min_power_path_using_dijkstra(self, source, destination): 
        parents = {node : None for node in self.nodes.values()}
        powers = {node : 0 if node == source else -1 for node in self.nodes.values()}
        
        dijkstra_with_power(source, parents, powers)

        power = powers[destination]
        path = get_path_from_parents(source, destination, parents)

        return path, power
    
    def get_min_power_path_from_MST(self, route, broke=True):
        source = route.source.value
        destination = route.destination.value
        path = self.MST.route_characteristics(source, destination, broke, path=True)

        return path

    def kruskal(self):
        new_graph, parent, rank = self.kruskal_initialisation()

        for edge in self.edges:
            edge_node1 = new_graph.nodes[edge.node1.value]
            edge_node2 = new_graph.nodes[edge.node2.value]

            new_edge = GraphEdge(edge_node1, edge_node2, edge.power, edge.distance)
            node1 = find(parent, edge_node1)
            node2 = find(parent, edge_node2)

            if node1 != node2:
                add_edge_to_Graph(new_graph, new_edge)
                union(parent, rank, node1, node2)

            if len(new_graph.edges) == len(self.nodes) - 1 :
                new_graph.sort_edges()

        self.MST = graph_to_tree(new_graph, self.station)
        
    def kruskal_initialisation(self):
        new_graph = Graph()
        parent = {}
        rank = {}

        for node in self.nodes.values():
            new_node = GraphNode(node.value)
            rank[new_node] = 0
            parent[new_node] = new_node
            new_graph.nodes[new_node.value] = new_node

        return new_graph, parent, rank

        
def get_path_from_parents(source, destination, parents):
    path = [destination.value]
    
    node = destination

    while node != source: 
        node = parents[node]  
        path.append(node.value)
    
    return path


def dijkstra_with_power(source, parents, powers):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()
        update_neighbours_power(node, heap, parents, powers)
            
def update_neighbours_power(node, heap, parents, powers):
    for neighbour, edges in node.neighbours.items():
        for edge in edges:
            if (powers[neighbour] == -1
                or max(edge.power, powers[node]) < powers[neighbour]):

                parents[neighbour] = node
                powers[neighbour] = max(edge.power, powers[node])

                if heap.have_wrap(neighbour):
                    heap.decrease_key(neighbour, powers[neighbour])

                else: 
                    heap.insertion(neighbour, powers[neighbour])
                                    

def dijkstra_with_distance(source, parents, distances, truck_power):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()
        update_neighbours_distance(node, heap, parents, distances, truck_power)

def update_neighbours_distance(node, heap, parents, distances, truck_power):
    for neighbour, edges in node.neighbours.items():
        for edge in edges:
            if (truck_power >= edge.power
                and (distances[neighbour] == -1 
                     or distances[neighbour] >= distances[node] + edge.distance)):
                
                parents[neighbour] = node
                distances[neighbour] = distances[node] + edge.distance

                if heap.have_wrap(neighbour):
                    heap.decrease_key(neighbour, distances[neighbour])
                    
                else:
                    heap.insertion(neighbour, distances[neighbour])
            

def find(parent, node):
    if parent[node] != node:
        parent[node] = find(parent, parent[node])

    return parent[node]
    
def union(parent, rank, node1, node2):
    if rank[node1] < rank[node2]:
        parent[node1] = node2

    elif rank[node1] > rank[node2]:
        parent[node2] = node1

    else:
        parent[node2] = node1
        rank[node1] += 1


def graph_to_tree(graph, station_value):
    starting_node = graph.nodes[station_value]
    distance = 0
    power = 0

    root = TreeNode(starting_node.value, power, distance)

    spanning_tree = Tree(root)
    spanning_tree.gas_price = graph.gas_price
    spanning_tree
    spanning_tree.nodes[root.value] = root

    stack = [(starting_node, root)]
    visited = set([starting_node])
    epsilon = graph.broke_probability

    while stack:
        node, tree_node = stack.pop()
        for neighbour in node.neighbours.keys():
            if neighbour not in visited:
                edge = node.neighbours[neighbour][0]
                child = TreeNode(neighbour.value, edge.power, edge.distance, parent=tree_node)

                if child.broke_with_probability(epsilon):
                    child.broke = True

                spanning_tree.nodes[neighbour.value] = child
                tree_node.children.append(child)
                stack.append((neighbour, child))
                visited.add(neighbour)

    return spanning_tree


def estimated_time_processing_using_dijkstra(graph):
    start = time.perf_counter()
    count = 0

    for route in graph.routes:
        graph.get_min_power_path_using_dijkstra(route.source, route.destination) 
        count += 1

        if count > 5:
            break 

    end = time.perf_counter()
    mean = (end-start)/10
    estimated_time = (mean * len(graph.routes))/3600
    
    return f"It will take around {estimated_time} hours processing."

def estimated_time_processing_from_MST(graph):
    start = time.perf_counter()
    count = 0

    for route in graph.routes:
        source_value = route.source.value
        destination_value = route.destination.value
        graph.MST.route_characteristics(source_value,  destination_value, broke=False)

        count += 1

        if count > 5:
            break 

    end = time.perf_counter()
    mean = (end-start)/5
    estimated_time = mean * len(graph.routes) 
    
    return f"It will take around {int(estimated_time)} seconds processing."             
     

def graph_from_file(network_filename, route_filename, station, gas_price, broke_probability):
    print("The graph is being created...")
    nb_nodes, edges_of_graph = open_network_file(network_filename)

    graph = Graph()
    graph.gas_price = gas_price
    graph.broke_probability = broke_probability
    graph.station = station

    set_nodes_to_graph(nb_nodes, graph)
    set_edges_to_graph(edges_of_graph, graph)
    graph.sort_edges()

    graph.kruskal()
    set_routes_to_graph(route_filename, graph)
    graph.routes.sort(key=lambda route: route.utility, reverse=True)

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

        node1 = graph.nodes[node1_value]
        node2 = graph.nodes[node2_value]
        
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

def set_routes_to_graph(route_file, graph):
    graph_routes = open_route_file(route_file)

    for line in graph_routes:
        route = get_data_from_line(line)
        node1_value, node2_value, utility = route

        node1 = graph.nodes[node1_value]
        node2 = graph.nodes[node2_value]
        
        new_route = GraphRoute(node1, node2, utility)
        graph.routes.append(new_route)
        graph.set_route_characteristics(new_route)

def open_route_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")

        routes_of_graph = lines[1:]

        return routes_of_graph


def create_displayable_network(graph):
    G = nx.Graph()

    for node_value in graph.nodes.keys():
        G.add_node(node_value)
    
    for edge in graph.edges:
        node1 = edge.node1.value
        node2 = edge.node2.value
        G.add_edge(node1, node2, distance=edge.distance)

    return G

def display_network(network, title):
    node_positions = nx.spring_layout(network, seed=42)

    plt.figure()
    nx.draw(network, node_positions, with_labels=True, node_color='lightblue', font_weight='bold')
    plt.savefig(title)
    plt.close()

def create_displayable_route(network, graph, route):
    path = graph.get_min_power_path_from_MST(route, broke=False)

    edges_from_path = []
    for n1, n2 in network.edges():
        if n1 in path and n2 in path:
            index1 =  path.index(n1)
            index2 = path.index(n2)
            
            if index1 == index2-1 or index2 == index1-1:
                edges_from_path.append((n1, n2))

    route = network.edge_subgraph(edges_from_path)

    return route

def display_route(network, route, title):
    node_positions = nx.spring_layout(network, seed=42)

    plt.figure()
    edge_labels = nx.get_edge_attributes(route, 'distance')
    nx.draw(route, node_positions, with_labels=True, node_color='lightblue', font_weight='bold')
    nx.draw_networkx_edge_labels(network, node_positions, edge_labels=edge_labels)
    plt.savefig(title)
