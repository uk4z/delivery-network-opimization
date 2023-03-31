import sys 
sys.path.append("D:\Coding files\Delivery network\Fibonacci Heap")
from fibonacci_heap import FibonacciHeap
import collections
import time


class DeliveryNetwork:
    def __init__(self, graph):
        self.graph = graph
        self.trucks = {}
        self.not_available_routes = set()

    def add_truck(self, truck):
        cost = truck.cost 

        if cost in self.trucks:
            self.trucks[cost].append(truck)
            self.trucks[cost].sort(key=lambda truck: truck.truck_power, reverse=True)
        
        else: 
            self.trucks[cost] = [truck]

    def set_profit_to_truck(self, truck):
        truck_power = truck.truck_power
        
        while truck_power >= 0:
            if truck_power in self.graph.powerRoutes:
                for route in self.graph.powerRoutes[truck_power]:
                    if route not in self.not_available_routes:
                        truck.profit = route.utility
                        self.not_available_routes.add(route)
                        return 
            truck_power -= 1

    def to_buy_with_budget(self, budget):
        already_computed_solutions = {}
        truck_costs = list(self.trucks.keys())
        truck_profits = []
        nb_trucks = len(self.trucks)

        for trucks in self.trucks.values():
            for truck in trucks:
                truck_profits.append(truck.profit)

        "Starting to get the highest profit..."

        knapsack(already_computed_solutions, budget, truck_costs, truck_profits, nb_trucks)
    
        return already_computed_solutions[nb_trucks]


def knapsack(already_computed_solutions, budget, truck_costs, truck_profits, nb_trucks):
    if nb_trucks == 0 or budget == 0:
        return 0

    if truck_costs[nb_trucks - 1] > budget:
        return knapsack(already_computed_solutions, budget, truck_costs, truck_profits, nb_trucks - 1)

    if nb_trucks in already_computed_solutions:
        return already_computed_solutions[nb_trucks]

    value_picked = truck_profits[nb_trucks - 1] + knapsack(already_computed_solutions, budget-truck_costs[nb_trucks - 1], truck_costs, truck_profits, nb_trucks - 1)
    value_notpicked = knapsack(already_computed_solutions, budget, truck_costs, truck_profits, nb_trucks - 1)
    value_max = max(value_picked, value_notpicked)

    already_computed_solutions[nb_trucks] = value_max

    return value_max


class Truck:
    def __init__(self, cost, truck_power):
        self.cost = cost 
        self.truck_power = truck_power
        self.profit = 0 


class TreeNode:
    def __init__(self, value, power, parent=None):
        self.parent = parent
        self.children = []
        self.value = value
        self.power = power


class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = {}

    def find_min_power(self, value1, value2):
        if value1 == value2:
            return -1
        
        node1 = self.nodes[value1]
        node2 = self.nodes[value2]
        lca = lowest_common_ancestor(node1, node2)
        min_power = 0

        node = node1
        while node != lca:
            min_power = max(min_power, node.power)
            node = node.parent 
        
        node = node2
        while node != lca:
            min_power = max(min_power, node.power)
            node = node.parent
    
        return  min_power


def lowest_common_ancestor(node1, node2):
    ancestors = set()

    while node1:
        ancestors.add(node1)
        node1 = node1.parent

    while node2:
        if node2 in ancestors:
            return node2
        node2 = node2.parent

    return None


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


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.powerRoutes = {}
        self.minimum_spanning_tree = None
        self.routes = []

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

    def add_minimum_spanning_tree(self):
        minimum_graph = kruskal(self)
        self.minimum_spanning_tree = graph_to_tree(minimum_graph)

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

    def associate_power_with_route(self, route):
        source = route.source.value
        destination = route.destination.value
        power = self.minimum_spanning_tree.find_min_power(source, destination)

        if power not in self.powerRoutes.keys():
            self.powerRoutes[power] = [route]

        else: 
            self.powerRoutes[power].append(route)

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
    
    def get_min_power_path_from_MST(self, source, destination):
        visited = set()
        min_power = -1

        return dfs(visited, source, destination, min_power)
    
    def get_min_powers_from_source(self, source):
        peres = {node : None for node in self.nodes.values()}
        powers = {node : -1 for node in self.nodes.values()}
        powers[source] = 0
        
        dijkstra_with_power(source, peres, powers)

        return powers
    


def dfs(visited, node, destination, min_power):
    if node == destination:
        return min_power
    
    visited.add(node)

    for neighbour, edges in node.neighbours.items():
        if neighbour not in visited:
            old_min_power = min_power
            min_power = max(min_power, edges[0].power)
            result = dfs(visited, neighbour, destination, min_power)
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


def graph_to_tree(graph):
    starting_node = graph.edges[0].node1
    root = TreeNode(starting_node.value, 0)
    spanning_tree = Tree(root)
    spanning_tree.nodes[root.value] = root
    stack = [(starting_node, root)]
    visited = set([starting_node])

    while stack:
        node, tree_node = stack.pop()
        for neighbour in node.neighbours.keys():
            if neighbour not in visited:
                edge = node.neighbours[neighbour][0]
                child = TreeNode(neighbour.value, edge.power, tree_node)
                spanning_tree.nodes[neighbour.value] = child
                tree_node.children.append(child)
                stack.append((neighbour, child))
                visited.add(neighbour)

    return spanning_tree


def deliveryNetwork_from_file(network_filename, route_filename, truck_filename):
    graph = graph_from_file(network_filename, route_filename)
    delivery_network = DeliveryNetwork(graph)

    set_truck_to_delivery_network(truck_filename, delivery_network)
    return delivery_network

def graph_from_file(network_filename, route_filename):
    print("The graph is being created...")
    nb_nodes, edges_of_graph = open_network_file(network_filename)
    graph = Graph()

    set_nodes_to_graph(nb_nodes, graph)
    set_edges_to_graph(edges_of_graph, graph)
    graph.sort_edges()

    graph.add_minimum_spanning_tree()
    set_routes_to_graph(route_filename, graph)
    graph.routes.sort(key=lambda route: route.utility, reverse=True)
    print("The graph has been created.")
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
        graph.associate_power_with_route(new_route)

def open_route_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")

        routes_of_graph = lines[1:]

        return routes_of_graph


def set_truck_to_delivery_network(truck_file, delivery_network):
    trucks_info = open_truck_file(truck_file)

    for line in trucks_info:
        truck = get_data_from_line(line)
        truck_power, cost = truck
        
        new_truck = Truck(truck_power, cost)
        delivery_network.add_truck(new_truck)
        delivery_network.set_profit_to_truck(new_truck)

def open_truck_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")

        trucks_info = lines[1:]

        return trucks_info


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

def estimated_time_processing_using_kruskal(graph):
    start = time.perf_counter()
    count = 0

    for route in graph.routes:
        graph.get_min_power_path_using_kruskal(route.source.value, route.destination.value) 
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
        graph.minimum_spanning_tree.find_min_power(route.source.value, route.destination.value)
        count += 1

        if count > 5:
            break 

    end = time.perf_counter()
    mean = (end-start)/5
    estimated_time = mean * len(graph.routes) 
    
    return f"It will take around {int(estimated_time)} seconds processing."             
                    

def write_min_power_in_output_file(filename, graph):
    print(estimated_time_processing_from_MST(graph))
    spanning_graph = kruskal(graph)
    tree = graph_to_tree(spanning_graph)

    with open(filename, "w") as file:
        for route in graph.routes:
            source = route.source
            destination = route.destination

            power = tree.find_min_power(source.value, destination.value)

            line = f"{source.value} {destination.value} {power}\n"
            file.write(line)

            


