from graph import *

network_filename = "input/network.2.in"
route_filename = "input/route.2.in"
truck_filename = "input/truck.2.in"

delivery_network = deliveryNetwork_from_file(network_filename, route_filename, truck_filename)

budget = 25*(10**9)

trucks_collection, profit = delivery_network.to_buy_with_budget(budget)

print(profit)
