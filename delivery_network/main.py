from graph import Graph


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

        node1 = graph.get_node_by_value(node1_value)
        node2 = graph.get_node_by_value(node2_value)
        
        new_edge = GraphEdge(node1, node2, power, distance)
        add_edge_to_Graph(graph, new_edge)

def add_edge_to_Graph(graph, edge):
    node1 = edge.node1
    node2 = edge.node2

    graph.edges.append(edge)

    node1.set_neighbor(node2, edge)
    node2.set_neighbor(node1, edge)
