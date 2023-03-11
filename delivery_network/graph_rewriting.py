def graph_from_file(filename):
    graph_nodes, graph_edges = open_file(filename)
    nodes = set_graph(graph_nodes)
    edges = set_edges(graph_edges, nodes)
    print(show_graph(edges))
    return nodes, edges

def open_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")
        graph_nodes = lines[0]
        graph_edges = lines[1:]
        return graph_nodes, graph_edges

def set_graph(graph_info):
    nodes = {}
    nb_nodes = get_data_from_line(graph_info)[0]
    for value in range(1, nb_nodes + 1):
        nodes[value] = Graph(value)
    return nodes 

def get_data_from_line(line):
    raw_data = line.split(' ')
    processed_data = [int(data) for data in raw_data]
    return processed_data

def set_edges(graph_edges, nodes):
    edges = []
    for line in graph_edges:
        edge = get_data_from_line(line)
        nb_info = len(edge)
        if nb_info == 4:
            node1_value, node2_value, power, distance = edge
        if nb_info == 3:
            node1_value, node2_value, power = edge
            distance = 1
        node1 = get_node(node1_value, nodes)
        node2 = get_node(node2_value, nodes)
        new_edge = add_edge(node1, node2, power, distance)
        edges.append(new_edge)
        
    return edges

def get_node(value, nodes):
    return nodes[value] 
    
def add_edge(node1=None, node2=None, power=0, distance=1):
    node1.set_neighbor(node2)
    node2.set_neighbor(node1)
    new_edge = Edge(node1, node2, power, distance)
    node1.edges.append(new_edge)
    node2.edges.append(new_edge)
    return new_edge

def show_graph(edges):
    if Graph.nb_nodes == 0:
        output = "The graph is empty."
    else:
        output = f"The graph has {Graph.nb_nodes} nodes and {Edge.nb_edges} edges.\n"
        for edge in edges:
            source = edge.node1.value
            destination = edge.node2.value
            output += f"{source} --- {destination}\n"
    
    return output

def connected_components_set(graph_nodes):
    output = set()
    for node in graph_nodes.values():
        connected_nodes = connected_components(node)
        if connected_nodes not in output:
            output.add(frozenset(connected_nodes))
    return output

def connected_components(node):
    visit = set() 
    def helper(node):
        if node.value in visit:
            return
        else:
            visit.add(node.value)
            for neighbor in node.neighbors.values():
                helper(neighbor)
    helper(node)
    return visit

class Graph:
    nb_nodes = 0

    def __init__(self, value=0, neighbors=None, edges=None):
        self.value = value
        self.neighbors = neighbors or {}
        self.edges = edges or []
        Graph.nb_nodes += 1
        self.peres = {}
        self.distances = {self : 0}
        self.already_processed = {self : True}
    
    def set_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors[node.value] = node 

    def shortest_route_with_power(self, destination, nodes): 
        dijkstra(self, nodes)
        end_to_start_route = [destination.value]
        distance = self.distances[destination]
        while destination != self: 
            pere = self.peres[destination]
            end_to_start_route.append(pere.value)
        start_to_end_route = route[::-1]
        return [start_to_end_route, distance]
        

def dijkstra(root):
    from collections import deque
    reach_nodes = deque()
    reach_nodes.append((0,root))
    while not reach_nodes.empty():
        sorted(reach_nodes, key=lambda tuple: tuple[0])
        node = reach_nodes.pop()[1]
        root.already_processed[node] = True
        visit_neighbors(root, node, reach_nodes)
    return root.peres, root.distances 

def find_neighbor_in_edge(node, edge):
    if edge.node1 != node:
        return edge.node1
    else: 
        return edge.node2
    
def uptade_priorityQueue(priorityQueue, element):
    priorityQueue.put(element)

def visit_neighbors(root, node, reach_nodes):
    for edge in node.edges:
        neighbor = find_neighbor_in_edge(node, edge)
        if (neighbor not in root.already_processed 
            and (neighbor not in root.distances 
                or root.distances[neighbor] > root.distances[node] + edge.distance)):
            root.peres[neighbor] = node
            root.distances[neighbor] = root.distances[node] + edge.distance
            reach_nodes.remove()
            reach_nodes.append(root.distances[neighbor], neighbor)

class Edge:
    nb_edges = 0

    def __init__(self, node1=None, node2=None, power=0, distance=1):
        self.node1 = node1
        self.node2 = node2
        self.power = power
        self.distance = distance
        Edge.nb_edges += 1
    
    def __str__(self):
        source = self.node1.value
        destination = self.node2.value
        return f"{source} --- {destination}"

network_filename = "input/network.00.in"

#graph_nodes, graph_edges = graph_from_file(network_filename)

#node = graph_nodes[1]
#print(node.shortest_route_with_power(graph_nodes[3], graph_nodes))

