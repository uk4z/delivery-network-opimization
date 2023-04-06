# ENSAE 1A: Programming Project

This is an object-oriented programming project which tries to replicate a delivery network. The network is composed by a collection of trucks and a representation of a map with cities and routes.

This repository contains several folders and files:

The `delivery_network` folder contains the main code. It is composed by 4 different files: `heap.py`, `tree.py`, `graph.py`, `delivery_network.py`.
  - `heap.py`: implementation of a Fibonacci heap used in `graph.py`
  - `tree.py`: implementation of a tree used in `graph.py`
  - `graph.py`: implementation of a graph used as a map in `delivery_network.py`
  - `delivery_network.py`: implementation of a delivery_network 
  
The `inputs` folder contains datasets (graphs, sets of routes and sets of trucks).

The `tests` folder contains unit tests.

## Input File Format

The input folder contains 3 types of files: 
  - network.x.in files ($x \in {00, 01, 02, 03, 04, 1, ..., 10}$) containing the graphs
  - routes.x.in files ($x \in {1, ..., 10}$) containing sets of routes for the corresponding x graphs.
  - trucks.x.in files ($x \in {1, ..., 3}$) containing sets of trucks.

### The structure of the network.x.in files is as follows:
  
<img src="https://user-images.githubusercontent.com/118286479/230010600-66a1d179-ccb3-43d8-8056-aad54a3c4e05.png" width="75" height="225">


`n` `m`  ---- number of vertices (n) and number of edges (m)

`city1` `city2` `power` `distance` ---- edge

where:
  - `city1`, `city2` are the vertices of the edge
  - `power` is the minimum required power to pass through the edge
  - `distance` (optional) is the distance between city1 and city2 on the edge

### The structure of the routes.x.in files is as follows:

<img src="https://user-images.githubusercontent.com/118286479/230011490-ea7d1375-8194-4233-a09e-5888f8f0e6bf.png" width="100" height="300">

`n` ---- number of routes

`city1` `city2` `utility` ---- route

where:
  - `city1`, `city2` are the source and destination of the route
  - `utility` is the profit gained if the corresponding route is covered

### The structure of the trucks.x.in files is as follows:

<img src="https://user-images.githubusercontent.com/118286479/230012685-e75172c2-90fb-4efe-9028-1894d79fb2a7.png" width="170" height="510">

`n` ---- number of trucks

`power` `cost` ---- truck

where:
  - `power` is the power of the truck
  - `cost` is the cost of the truck
  
## Delivery network File

Here, the code will be explained with particular interest on well known algorithm adapted for the purpose of this project. 

### Graph

The graph file is composed of a main class `Graph` and its 3 subclasses `GraphNode`, `GraphEdge`, `GraphRoute`. Furthermore, some functions are defined outside classes either being requested in class methods, participating to the construction of the graph or allowing the graph to be displayed. 

In the network, each vertex is associated to a unique value which will be used to identify the vertex in different data structures and classes. 

Below is a representation of *network.1.in* where *5* is the network station.

<img src="https://user-images.githubusercontent.com/118286479/230067770-00be6629-21b7-4333-917d-609798885416.png" width="400" height="300">


***Attributes description***

- `GraphEdge` represents a given edge of the network. It contains the two vertices of the edge (`node1` and `node2`), its `power` and `distance` as attributes. 
- `GraphNode` represents a given vertex of the network. It contains the vertex `value` and its `neighbours`. The neighbours attribute is a dictionnary with vertex as keys and a list of every edges linking the two vertices as values.   
- `GraphRoute` represents a given route of the network. It contains the two vertices (`source` and `destination`) of the route as well as its `utility` and mark it as `available` when a new route is created. 3 other attributes are initialised such as `power`, `cost` and `expected_utility`. `power` is the required power to complete the route. `cost` represents the price linked to the gas consumption if the route is covered. `expected_utility` caracterises the expected profit when covering the route considering that the route can be blocked with a certain probability.
- `Graph` contains all the information about the vertices, edges and routes which define the network in the attributes `nodes`, `edges`, `routes`. The `station` is also represented. `gas_price` defines the price per kilometer of gas consumption. Moreover, each edge has a certain probability to broke making it unable to use represented with `broke_probability` attribute. Last, `MST` attribute assign the minimum spanning tree (MST) of the graph weighted with power. The method to get the *MST* will be explained further later on. 

***Methods description***

Here not all methods will be covered because some of them are not particularly interesting and only participates to build a consistent network. Mainly, 2 methods will be overviewed: `get_path_given_power` and `kruskal`. 

-`get_path_given_power` allows the user to get the shortest path in the network from a source to a destination given the power of a truck. Therefore, `source`, `destination` and `truck_power` compose the arguments of this method. A default value of *float("inf")* is set to `truck_power` because the method should also work if we only want to find the shortest route between two vertices of the network without considering the power required. This method is an adaptation of *Dijksra's algorithm*. 


```ruby
def get_path_given_power(self, source, destination, truck_power=float("inf")):
    parents = {node : None for node in self.nodes.values()}
    distances = {node: 0 if node == source else -1 for node in self.nodes.values()}

    if not source.is_connected_with_power(destination, truck_power):
        return None

    dijkstra_with_distance(source, parents, distances, truck_power)

    distance = distances[destination]
    path = get_path_from_parents(source, destination, parents)

    return path, distance 
```
```ruby
def dijkstra_with_distance(source, parents, distances, truck_power):
    heap = FibonacciHeap()
    heap.insertion(source, 0)

    while heap.min_node:
        node = heap.extract_min()
        update_neighbours_distance(node, heap, parents, distances, truck_power)
```

```ruby
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
```

This adaptation of *Dijkstra's algorithm* uses a priority queue data structure which takes form as a *Fibonacci Heap*. This allows the decrease_key method to run in **O(1)** time complexity and the extract_min method in **O(log(n))** time complexity (where *n* represents the number of nodes in the heap). A full implementation is available in the `heap.py` file and will be discuss further in the according section.

Therefore, this algorithm runs in **O(|E| + |V|log|V|)** (where *|E|* represents the number of edges and *|V|* the number of vertices) time complexity. In the method, `distances`, `parents` and `heap` variables have a **O(|V|)** memory complexity which defines that of `get_path_given_power`.

Unlike common presentations of *Dijkstra's algorithm*, this adaptation start with a priority queue that contains only one item, and insert new items as they are discovered (decrease the key if it is in the queue, otherwise insert it). This allows to maintain a smaller priority queue in practice, speeding up the queue operations.

This implementation of `get_path_given_power` can be easily adapted to create a new method finding the path which uses the minimum power. Such method is called `get_min_power_path_using_Dijkstra` available in the `graph.py` file. The full method implementation will not be discussed further but runs with the same complexites as `get_path_given_power`.

As part of the delivery network optimization, the goal is to find the path with minimum power between two vertices because it unables to use trucks with lower cost. However, networks such as *network.4.in* have up to 200 000 vertices and 300 000 edges, hence the adaptation of *Dijkstra's algorithm* is no longer the best option to consider. A better option is to optimize the graph using *Kruskal's algorithm*.

-`kruskal` finds the mininum spanning tree from the graph and update the `MST` attribute. The tree is an instance of the `Tree` class and the method to get the path with minimum power will be explained in the according section. No arguments are required in this method because it works from the graph. The method is an adaptation of *Kruskal's algorithm* working in two parts. First, it uses the algorithm to create a graph with only the edges which will be used in the tree. Then, it creates the tree from the new graph.  

```ruby
def kruskal(self):
  new_graph, parent, rank = self.kruskal_initialisation()
  new_graph.broke_probability = self.broke_probability

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
```

```ruby
def find(parent, node):
  if parent[node] != node:
      parent[node] = find(parent, parent[node])

  return parent[node]
```

```ruby
def union(parent, rank, node1, node2):
  if rank[node1] < rank[node2]:
      parent[node1] = node2

  elif rank[node1] > rank[node2]:
      parent[node2] = node1

  else:
      parent[node2] = node1
      rank[node1] += 1

```

As an adaptation of *Kruskal's algorithm*, the method is a *Union-Find* algorithm which allows to reach the highest ancestor with the `find` method and to unify two branches of the tree with the `union` method. The edges list is already sorted during the creation of the graph, therefore the method doesn't require to sort the list of edges before going through each edge. 

*Kruskal's algorithm* can be shown to run in **O(|E|log|V|)** time complexity,  all with simple data structures. However, the `graph_to_tree` call will add another **O(|V|)** time complexity as it browses every vertices from the new graph to create the tree. Hence, the overall time complexity is **O(|V| + |E|log|V|)**. 

Time complexity comparison between *Kruskal's algorithm* and *Dijkstra's algorithm*:


### Tree

<img src="https://user-images.githubusercontent.com/118286479/230014823-cb4ccced-80b1-4272-90a4-e054db30d53f.png" width="129" height="147">

This is a well known data structure in python. Such tree is
