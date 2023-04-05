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

Below is a representation of *network.1.in*  

<img src="https://user-images.githubusercontent.com/118286479/230018342-8f06984e-c095-4082-8037-bc1e246565cd.png" width="400" height="300">

- `GraphEdge` represents a given edge of the network. It contains the two vertices of the edge (`node1` and `node2`), its `power` and `distance` as attributes. 
- `GraphNode` represents a given vertex of the network. It contains the vertex `value` and its `neighbours`. The neighbours attribute is a dictionnary with vertex as keys and a list of every edges linking the two vertices as values.   
- `GraphRoute` represents a given route of the network. It contains the two vertices (`source` and `destination`) of the route as well as its `utility` and mark it as `available` when a new route is created. 3 other attributes are initialised such as `power`, `cost` and `expected_utility`. `power` is the required power to complete the route. `cost` represents the price linked to the gas consumption if the route is covered. `expected_utility` caracterises the expected profit when covering the route considering that the route can be blocked with a certain probability.

### Tree

<img src="https://user-images.githubusercontent.com/118286479/230014823-cb4ccced-80b1-4272-90a4-e054db30d53f.png" width="129" height="147">

This is a well known data structure in python. Such tree is
