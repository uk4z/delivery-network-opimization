from delivery_network import *
               

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




network_filename = "input/network.1.in"
route_filename = "input/route.1.in"
truck_filename = "input/truck.2.in"

delivery_network = deliveryNetwork_from_file(network_filename, route_filename, truck_filename)
budget = 25*(10**9)


plot_collections(delivery_network, budget)
