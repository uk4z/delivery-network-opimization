import random 
from graph import *

class Truck:
    def __init__(self, cost, truck_power):
        self.cost = cost 
        self.power = truck_power
        self.available = True
        self.profit = 0 


class DeliveryNetwork:
    def __init__(self, graph):
        self.graph = graph
        self.trucks = []

    def set_profit_to_truck(self):
        nb_trucks = len(self.trucks)
        nb_routes = len(self.graph.routes)
        truck_index = 0
        route_index = 0

        while truck_index < nb_trucks and route_index < nb_routes:
            route = self.graph.routes[route_index]
            index = truck_index
            optimized_truck_index = None

            while index < nb_trucks:
                truck = self.trucks[index]

                if truck.power < route.power:
                    break
                
                optimized_truck_index = index if truck.available else optimized_truck_index
                index += 1
            
            route.available = False if optimized_truck_index is not None else True
            route_index += 1

            if optimized_truck_index is None:
                continue
            
            else:
                truck = self.trucks[optimized_truck_index] 
                
                truck.profit = route.utility
                truck.available = False

                if optimized_truck_index == truck_index:
                    truck_index += 1
            
    def to_buy_with_budget(self, budget):
        mutation_rate = 0.1
        nb_solutions = 15
        running_time = 10
        print(f"Getting the optimal collection of trucks will take around {running_time} seconds.")

        if len(self.trucks) > len(self.graph.routes):
            trucks = []
            for i in range(len(self.trucks)):
                truck = self.trucks[-(i+1)]
                if not truck.available:
                    trucks.append(truck)

        else: 
            trucks = self.trucks

        solution , profit = genetic_algorithm(trucks, budget, nb_solutions, mutation_rate, running_time )    
        set_of_trucks = []
        
        for i in range(len(solution)):
            if solution[i] == 1:
                set_of_trucks.append(self.trucks[i])
        
        return set_of_trucks, profit


def create_random_solution(size):
    solution = []
    for i in range(0, size):
        solution.append(random.randint(0, 1))
    return solution

def valid_solution(trucks, solution, budget):
    nb_trucks = len(trucks)
    total_cost = 0

    for i in range(nb_trucks):
        if solution[i] == 1:
            total_cost += trucks[i].cost

            if total_cost > budget:
                return False
        
    return True

def calculate_profit(trucks, solution):
    nb_trucks = len(trucks)
    total_profit = 0

    for i in range(nb_trucks):
        if solution[i] == 1:
            total_profit += trucks[i].profit
        
    return total_profit

def initial_solutions(pop_size, trucks, budget):
    solutions = []
    i = 0
    while i < pop_size:
        new_solution = create_random_solution(len(trucks))

        if valid_solution(trucks, new_solution, budget):
            if new_solution not in solutions:
                solutions.append(new_solution)
                i += 1
            
    return solutions

def tournament_selection(trucks, solutions):
    pop_size = len(solutions)
    id1 = random.randint(0, pop_size - 1)
    id2 = random.randint(0, pop_size - 1)

    solution1 = solutions[id1]
    solution2 = solutions[id2]

    profit1 = calculate_profit(trucks, solution1)
    profit2 = calculate_profit(trucks, solution2)

    if profit1 > profit2:
        return solution1
    
    else: 
        return solution2

def crossover(solution1, solution2, trucks, budget):
    while True:
        break_point = random.randint(0, len(solution1))
        first_part = solution1[:break_point]
        second_part = solution2[break_point:]
        new_solution = first_part + second_part

        if valid_solution(trucks, new_solution, budget):
            return new_solution

def mutation(solution, trucks, budget):
    while True: 
        mutated_solution = solution
        index_1, index_2 = random.sample(range(0, len(solution)), 2)
        mutated_solution[index_1], mutated_solution[index_2] = mutated_solution[index_2], mutated_solution[index_1]

        if valid_solution(trucks, mutated_solution, budget):
            return mutated_solution

def create_new_solutions(trucks, solutions, mut_rate, budget):
    new_solutions = []
    for i in range(0, len(solutions)):
        solution1 = tournament_selection(trucks, solutions)
        solution2 = tournament_selection(trucks, solutions)
        solution = crossover(solution1, solution2, trucks, budget)

        if random.random() <= mut_rate:
            solution = mutation(solution, trucks, budget)

        new_solutions.append(solution)
    return new_solutions

def best_solution(solutions, trucks):
    max_profit = 0
    best_solution = []
    for solution in solutions:
        profit = calculate_profit(trucks, solution)
        if profit > max_profit:
            max_profit = profit
            best_solution = solution

    return best_solution, max_profit

def genetic_algorithm(trucks, budget, nb_solutions, mutation_rate, running_time):
    solutions = initial_solutions(nb_solutions, trucks, budget)
    profits = {}

    timer = 0
    ref_time = time.time()
    while timer <= running_time:
        solutions = create_new_solutions(trucks, solutions, mutation_rate, budget)
        solution, profit = best_solution(solutions, trucks)
        profits[profit] = solution
        
        current_time = time.time()
        timer = round(current_time-ref_time)

    max_profit = max(profits.keys())

    return profits[max_profit], max_profit


def deliveryNetwork_from_file(network_filename, route_filename, truck_filename):
    graph = graph_from_file(network_filename, route_filename)
    delivery_network = DeliveryNetwork(graph)

    set_truck_to_delivery_network(truck_filename, delivery_network)
    delivery_network.trucks.sort(key=lambda truck: truck.power, reverse=True)

    print("Starting to assigned profits to trucks...")

    delivery_network.set_profit_to_truck()

    print("Delivery network is ready.")


    return delivery_network


def set_truck_to_delivery_network(truck_file, delivery_network):
    trucks_info = open_truck_file(truck_file)

    i = 0
    for line in trucks_info:
        i += 1
        truck = get_data_from_line(line)
        truck_power, cost = truck
        
        new_truck = Truck(truck_power, cost)
        delivery_network.trucks.append(new_truck)

def open_truck_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")

        trucks_info = lines[1:]

        return trucks_info


def plot_collections(delivery_network, budget):
    trucks_collection, profit = delivery_network.to_buy_with_budget(budget)

    network = create_displayable_network(delivery_network.graph)
    display_network(network, "graph")

    routes = []
    for route in delivery_network.graph.routes:
        if not route.available:
            source_value = route.source.value
            destination_value = route.destination.value

            dis_route = create_displayable_route(network, delivery_network.graph, source_value, destination_value)
        
            routes.append(dis_route)

    for route in routes:
        node_positions = nx.spring_layout(network, seed=42)

        nx.draw(route, node_positions, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.savefig("path")

    plt.close()
