from graph import *
import random 

class Truck:
       def __init__(self, cost, truck_power):
        self.cost = cost 
        self.power = truck_power
        self.routes = []


class DeliveryNetwork:
    def __init__(self, graph):
        self.graph = graph
        self.trucks = []
        self.station = None
        self.gas_price = 0

    def set_route_to_truck(self, budget):
        total_cost = 0

        for route in self.graph.routes:
            truck_id = self.select_truck_id(route)
            
            if truck_id is not None:
                truck = self.trucks[truck_id] 
                
                if total_cost + truck.cost > budget:
                    break

                route.available = False
                total_cost += truck.cost
                truck.routes.append(route)
            
    def select_truck_id(self, route):
        nb_type_trucks = len(self.trucks)
        optimized_truck_id = None

        if route.available:
            truck_id = 0
            
            
            while truck_id < nb_type_trucks:
                truck = self.trucks[truck_id]

                if truck.power < route.power:
                    break
                
                optimized_truck_id = truck_id
                truck_id += 1

        return optimized_truck_id

    def pre_selection_of_trucks(self):
        last_selected_truck = self.trucks[0]
        trucks = [last_selected_truck]

        for truck in self.trucks:
            if truck.cost < last_selected_truck.cost:
                trucks.append(truck)
                last_selected_truck = truck
            
        self.trucks = trucks 

    def to_buy_with_budget(self, budget):
        mutation_rate = 0.5
        nb_solutions = 15
        running_time = 30

        print(f"Getting the optimal collection of trucks will take around {running_time} seconds.")
        expected_gain = self.get_expected_gain()
        trucks, size = self.trucks_initialisation()

        if not trucks:
            return [], expected_gain, expected_gain

        solution, gain = genetic_algorithm(trucks, budget, nb_solutions, mutation_rate, running_time, size)
        total_gas_cost = calculate_cost(trucks, solution)
        
        
        profit = int(gain - total_gas_cost)
        return trucks, profit, expected_gain
    
    def trucks_initialisation(self):
        size = 0
        trucks = []
        for truck in self.trucks:
            nb_routes = len(truck.routes)
            size += nb_routes
            if nb_routes != 0:
                trucks += [truck]

        return trucks, size

    def get_expected_gain(self):
        expected_gain = 0 

        for route in self.graph.routes:
            expected_gain += route.expected_utility
        
        return int(expected_gain)
    

def create_random_solution(size):
    solution = []
    for _ in range(0, size):
        solution.append(random.randint(0, 1))
    return solution

def valid_solution(trucks, solution, budget):
    total_cost = 0
    route_count = 0

    for truck_type in trucks:
        nb_routes = len(truck_type.routes)
        total_cost += sum([truck_type.routes[k].cost for k in range(nb_routes) if solution[route_count+k]==1])
        route_count += nb_routes

        if total_cost > budget:
            return False
        
    return True

def calculate_profit(trucks, solution):
    total_profit = 0
    route_count = 0

    for truck_type in trucks:
        nb_routes = len(truck_type.routes)
        total_profit += sum([truck_type.routes[k].utility for k in range(nb_routes) if solution[route_count+k]==1])
        route_count += nb_routes

    return total_profit

def calculate_cost(trucks, solution):
    total_cost = 0
    route_count = 0

    for truck_type in trucks:
        nb_routes = len(truck_type.routes)
        total_cost += sum([truck_type.routes[k].cost for k in range(nb_routes) if solution[route_count+k]==1])
        route_count += nb_routes

    return total_cost

def initial_solutions(pop_size, trucks, budget, solution_size):
    solutions = []
    i = 0
    while i < pop_size:
        new_solution = create_random_solution(solution_size)

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

    if profit1 >= profit2:
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
    for _ in range(0, len(solutions)):
        solution1 = tournament_selection(trucks, solutions)
        solution2 = tournament_selection(trucks, solutions)
        solution = crossover(solution1, solution2, trucks, budget)

        if random.random() <= mut_rate:
            solution = mutation(solution, trucks, budget)

        new_solutions.append(solution)
    return new_solutions

def best_solution(solutions, trucks):
    max_profit = 0
    best_solution_profit = 0
    best_solution = []
    for solution in solutions:
        profit = calculate_profit(trucks, solution)
        if profit > max_profit:
            max_profit = profit
            best_solution = solution
            best_solution_profit = profit

    return best_solution, best_solution_profit

def genetic_algorithm(trucks, budget, nb_solutions, mutation_rate, running_time, solution_size):
    solutions = initial_solutions(nb_solutions, trucks, budget, solution_size)
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

    solution = profits[max_profit]
    return solution, max_profit


def deliveryNetwork_from_file(network_filename, route_filename, truck_filename, station, gas_price, broke_probability, budget):
    graph = graph_from_file(network_filename, route_filename, station, gas_price, broke_probability)
    delivery_network = DeliveryNetwork(graph)
    delivery_network.station = station
    delivery_network.gas_price = gas_price

    set_truck_to_delivery_network(truck_filename, delivery_network)
    delivery_network.trucks.sort(key=lambda truck: truck.power, reverse=True)
    delivery_network.pre_selection_of_trucks()

    print("Starting to assigned profits to trucks...")

    delivery_network.set_route_to_truck(budget)

    print("Delivery network is ready.")


    return delivery_network


def set_truck_to_delivery_network(truck_file, delivery_network):
    trucks_info = open_truck_file(truck_file)

    i = 0
    for line in trucks_info:
        i += 1
        truck = get_data_from_line(line)
        truck_power, cost = truck
        
        new_truck = Truck(cost, truck_power)
        delivery_network.trucks.append(new_truck)

def open_truck_file(filename):
    with open(filename, 'r') as file:
        input = file.read()
        lines = input.split("\n")

        trucks_info = lines[1:]

        return trucks_info


def plot_collections(delivery_network, budget):
    delivery_network.to_buy_with_budget(budget)

    network = create_displayable_network(delivery_network.graph)
    display_network(network, delivery_network.graph.station, "graph")

    routes = []
    for truck in delivery_network.trucks:
        for route in truck.routes:
            dis_route = create_displayable_route(network, delivery_network.graph, route)
            routes.append(dis_route)

    for route in routes:
        node_positions = nx.spring_layout(network, seed=42)

        nx.draw(route, node_positions, with_labels=True, node_color="lightblue", font_weight='bold')
        plt.savefig("path")

    plt.close()
