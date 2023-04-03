from fibonacci_heap import FibonacciHeap
from tree import *

import collections

import time

import networkx as nx
import matplotlib.pyplot as plt


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

    def sort_edges_in_neighbours(self):
        for neighbour, edges in self.neighbours.items():
            sorted_edges = sorted(edges, key=lambda edge: edge.power, reverse=True)
            self.neighbours[neighbour] = sorted_edges

    def is_connected_with_power(self, target, truck_power=float("inf")):
        visited = set()
        stack = [self]

        while stack:
            node = stack.pop()

            if node == target:
                return True
            
            visited.add(node)
                
            for neighbour, edges in node.neighbours.items():
                if neighbour not in visited:
                    for edge in edges:
                        if truck_power >= edge.power:
                            stack.append(neighbour)
                            break

        return False


class GraphRoute:
    def __init__(self, source, destination, utility):
        self.source = source
        self.destination = destination
        self.utility = utility 
        self.cost = 0
        self.expected_utility = 0
        self.available = True
        self.power = 0


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.minimum_spanning_tree = None
        self.routes = []
        self.gas_price = 0

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

    def add_minimum_spanning_tree(self, station):
        minimum_graph = kruskal(self)
        self.minimum_spanning_tree = graph_to_tree(minimum_graph, station)

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
            if len(result) == len(self.nodes):
                return result 
            
            if node.value not in result:
                connect = self.connected_components(node.value)
                result.add(frozenset(connect))

        return result
    
    def sort_edges(self):
        self.edges.sort(key=lambda edge: edge.power)
        for node in self.nodes.values():
            node.sort_edges_in_neighbours()

    def associate_power_with_route(self, route, broke_routes=True): 
        source = route.source.value
        destination = route.destination.value
        epsilon = 0.001
        power, route_cost  = self.minimum_spanning_tree.find_min_power(source, destination, self.gas_price, broke_routes)
        nb_edges = self.minimum_spanning_tree.nb_edges_in_route(source, destination)
        probability_of_success = (1-epsilon)**nb_edges
        probability_of_failure = 1-probability_of_success

        route.expected_utility = route.utility*(probability_of_success - probability_of_failure)

        if power is None:
            route.utility = 0
            route.power = 0

        else:
            route.power = power       
        
        route.cost = route_cost    
        


    def get_path_given_power(self, source_value, destination_value, truck_power=float("inf")):
        source = self.nodes[source_value]
        destination = self.nodes[destination_value]

        if not source.is_connected_with_power(destination, truck_power):
            return None
        
        path, distance = self._shortest_path(source, destination, truck_power)

        if distance !=  -1:
            return path, distance 
        
        else:
            path, distance = self._shortest_path(destination, source, truck_power)
            new_path = collections.deque()

            for node in path:
                new_path.appendleft(node)

            return new_path, distance

    def _shortest_path(self, source, destination, truck_power): 
        peres = {node : None for node in self.nodes.values()}
        distances = {node : -1 for node in self.nodes.values()}
        distances[source] = 0
        
        dijkstra_with_distance(source, peres, distances, truck_power)

        distance = distances[destination]
        path = get_path_from_peres(source, destination, peres)

        return path, distance 
    
    def get_min_power_path_using_dijkstra(self, source, destination): 
        peres = {node : None for node in self.nodes.values()}
        powers = {node : -1 for node in self.nodes.values()}
        powers[source] = 0
        
        dijkstra_with_power(source, peres, powers)

        power = powers[destination]
        path = get_path_from_peres(source, destination, peres)

        return path, power
    
    def get_min_power_path_from_MST(self, route, broke_routes=True):
        source = route.source.value
        destination = route.destination.value
        path, route_cost = self.minimum_spanning_tree.find_min_power_path(source, destination, self.gas_price, broke_routes)

        if path is None:
            route.utility = 0

        return path, route_cost


def dfs_min_power(visited, node, destination, min_power):
    if node == destination:
        return min_power
    
    visited.add(node)

    for neighbour, edges in node.neighbours.items():
        if neighbour not in visited:
            old_min_power = min_power
            min_power = max(min_power, edges[0].power)
            result = dfs_min_power(visited, neighbour, destination, min_power)
            if result != -1:
                return result
            else: 
                min_power = old_min_power

    return -1


def dijkstra_with_power(source, peres, powers):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()

        update_neighbours_power(node, heap, peres, powers)
            
def update_neighbours_power(node, heap, peres, powers):
        for edges in node.neighbours.values():
            for edge in edges:
                neighbour = edge.find_neighbour_in_edge(node)

                if (powers[neighbour] == -1
                    or max(edge.power, powers[node]) < powers[neighbour]):
                    peres[neighbour] = node
                    powers[neighbour] = max(edge.power, powers[node])

                    if heap.have_wrap(neighbour):
                        heap.decrease_key(neighbour, powers[neighbour])

                    else: 
                        heap.insertion(neighbour, powers[neighbour])
                                    

def dijkstra_with_distance(source, peres, distances, truck_power):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()

        update_neighbours_distance(node, heap, peres, distances, truck_power)

def update_neighbours_distance(node, heap, peres, distances, truck_power):
        for edges in node.neighbours.values():
            for edge in edges:
                if truck_power >= edge.power:
                    neighbour = edge.find_neighbour_in_edge(node)

                    if (distances[neighbour] == -1 
                        or distances[neighbour] >= distances[node] + edge.distance):
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


def kruskal(graph):
    new_graph = Graph()
    parent = {}
    rank = {}

    for edge in graph.edges:
        first_value = edge.node1.value
        if first_value not in new_graph.nodes:
            node = GraphNode(first_value)
            new_graph.nodes[first_value] = node 
            rank[node] = 0 
            parent[node] = node

        second_value = edge.node2.value
        if second_value not in new_graph.nodes:
            node = GraphNode(second_value)
            new_graph.nodes[second_value] = node
            rank[node] = 0
            parent[node] = node

        new_edge = GraphEdge(new_graph.nodes[edge.node1.value], new_graph.nodes[edge.node2.value], edge.power, edge.distance)
        node1 = find(parent, new_graph.nodes[edge.node1.value])
        node2 = find(parent, new_graph.nodes[edge.node2.value])

        if node1 != node2:
            add_edge_to_Graph(new_graph, new_edge)
            union(parent, rank, node1, node2)

        if len(new_graph.edges) == len(graph.nodes) - 1 :
            new_graph.sort_edges()
            return new_graph
    
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
    spanning_tree.nodes[root.value] = root

    stack = [(starting_node, root)]
    visited = set([starting_node])
    epsilon = 0.001

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
        graph.get_min_power_path_using_dijkstra(route.source.value, route.destination.value) 
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
        broke_routes = False
        graph.minimum_spanning_tree.find_min_power(route.source.value, route.destination.value, broke_routes)

        count += 1

        if count > 5:
            break 

    end = time.perf_counter()
    mean = (end-start)/5
    estimated_time = mean * len(graph.routes) 
    
    return f"It will take around {int(estimated_time)} seconds processing."             
     

def graph_from_file(network_filename, route_filename, station, gas_price):
    print("The graph is being created...")
    nb_nodes, edges_of_graph = open_network_file(network_filename)
    graph = Graph()
    graph.gas_price = gas_price

    set_nodes_to_graph(nb_nodes, graph)
    set_edges_to_graph(edges_of_graph, graph)
    graph.sort_edges()

    graph.add_minimum_spanning_tree(station)
    set_routes_to_graph(route_filename, graph, gas_price)
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

def set_routes_to_graph(route_file, graph, gas_price):
    graph_routes = open_route_file(route_file)

    for line in graph_routes:
        route = get_data_from_line(line)
        node1_value, node2_value, utility = route

        node1 = graph.nodes[node1_value]
        node2 = graph.nodes[node2_value]
        
        new_route = GraphRoute(node1, node2, utility)
        graph.routes.append(new_route)
        graph.associate_power_with_route(new_route, gas_price)

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
    path_nodes = graph.get_min_power_path_from_MST(route, broke_routes=False)[0]

    edges_from_path_nodes = []
    for n1, n2 in network.edges():
        if n1 in path_nodes and n2 in path_nodes:
            index1 =  path_nodes.index(n1)
            index2 = path_nodes.index(n2)
            
            if index1 == index2-1 or index2 == index1-1:
                edges_from_path_nodes.append((n1, n2))

    route = network.edge_subgraph(edges_from_path_nodes)

    return route

def display_route(network, route, title):
    node_positions = nx.spring_layout(network, seed=42)

    plt.figure()
    edge_labels = nx.get_edge_attributes(route, 'distance')
    nx.draw(route, node_positions, with_labels=True, node_color='lightblue', font_weight='bold')
    nx.draw_networkx_edge_labels(network, node_positions, edge_labels=edge_labels)
    plt.savefig(title)
