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
  
## Delivery network Files

Here, the code will be explained with particular interest on well known algorithm adapted for the purpose of this project. In the network, each vertex is associated to a unique value which will be used to identify the vertex in different data structures and classes. 

### Fibonacci heap

The `heap.py`file contains the implementation of a *Fibonacci heap*. This heap data structure is the most efficient priority queue especially with `decrease_key` and `extract_min` methods. For the purpose of this project, the *Fibonacci heap* will be used in an adapatation of *Dijkstra's algorithm* to find the shortest route between two vertices in the graph. 

The file is composed of a main class `FibonacciHeap` and its subclass `FibonacciHeapNode`. A function is defined outside classes and is called by methods. 

In such data structure, a pointer is set to the node with the minimum key which is part of the roots list. Working as a heap data structure, every children have higher keys than their parents.

<img src="https://user-images.githubusercontent.com/118286479/230576019-72241f07-8422-4e60-ac2c-58411cd041ef.png" width="600" height="300">

***Attributes description***

-`FibonacciHeapNode` represents a node in the heap. Therefore as it is a priority queue data structure, `key` and `wrap` attributes are set to the class. An interesting point to observe is that `wrap` can be anything like a string, a value or even a node of the graph. As explained above, each node has its own `child` and `parent`. Moreover,  the current implementation does not require to consider a list of children but only to keep tracks of the closest neighbours. Therefore, each node has a pointer `left` and a pointer `right` (the representation of the heap below shows the link between each nodes). The attributes `degree`and `mark` are used in `decrease_key` and `consolidate` methods. The `degree` attribute caracterises the depth of a node in the heap and `mark` indicates if a parent node has lost one of its child.  

<img src="https://user-images.githubusercontent.com/118286479/230621885-da87df87-6d37-472b-9145-6725d57a2962.png" width="600" height="300">

-`FibonacciHeap` represents the heap. A pointer is set to the `min_node` which means the node with the lowest key and keeps track of the `root_list` which contains all nodes without parents. Also the attribute `nb_nodes` stores the number of nodes in the heap as it changes with each call of `insertion` and `extract_min`. 

***Methods description***

We will consider only two methods `decrease_key` and `extract_min` as they are used in an adaptation of *Dijkstra's algorithm*. The goal here is to express the time complexity of both methods because it will impact the overall complexity of *Dijkstra's algorithm*.

-`decrease_key` is a method to decrease the key of a given node already set in the heap. Depending of the how decreased the key is, it will impact its position in the heap. If the heap property becomes violated (the new key is smaller than the key of the parent), the node is `cut` from its parent.

```ruby
def decrease_key(self, wrap, new_key):
    node = self._get_node_by_wrap(wrap)

    if new_key >= node.key:
        return 

    node.key = new_key
    parent = node.parent

    if (parent is not None) and node.key < parent.key:
        self._cut(node, parent)
        self._cascade_cut(parent)

    if node.key < self.min_node.key:
        self.min_node = node
```

```ruby 
def _cut(self, node, parent):
    parent.remove_from_child_list(node)
    parent.degree -= 1
    self.merge_with_root_list(node)
    node.mark = False 
    node.parent = None
```

```ruby
def _cascade_cut(self, node):
    parent = node.parent

    if parent is not None:
        if not node.mark :
            node.mark = True

        else:
            self._cut(node, parent)
            self._cascade_cut(parent)
```

In the operation `decrease_key`, the parent is marked if it does not belong to the root list. If it has been marked already, it is cut as well and its parent is marked. We continue upwards until we reach either the root or an unmarked node (`cascade_cut`).  In the process say k nodes are added to the root list. The actual time to perform the cutting was **$O(k)$**, therefore the amortized running time is constant which means **$O(1)$**.

-`extract_min` deletes and returns the `min_node` of the heap. Therefore, the heap has to be updated with a new `min_node` and the structure will also change a bit. The calls of the method `consolidate` is here to update the structure of the tree. To resume, `extract_min` operates in three phases: remove `min_node`, `consolidate` the heap and find the new node with minimum key.

```ruby
def extract_min(self):
    extract_node = self.min_node

    if extract_node is not None:
        if extract_node.child is not None:
            child = extract_node.child

            while True:
                other_child = child.right
                self.merge_with_root_list(child)
                child.parent = None

                if other_child == extract_node.child:
                    break

                child = other_child

        self.remove_from_root_list(extract_node)

        if extract_node == extract_node.right:
            self.min_node = self.root_list = None

        else:
            self.min_node = extract_node.right
            self.consolidate()

        self.nb_nodes -= 1

        return extract_node.wrap
```

```ruby
def consolidate(self):
    MAX_DEGREE = 2 * int(math.log2(self.nb_nodes)) + 1
    root_list = [None] * (MAX_DEGREE + 1)

    node = self.min_node
    while True:
        degree = node.degree

        while root_list[degree] is not None:
            neighbour = root_list[degree]

            if neighbour.key < node.key:
                neighbour, node = node, neighbour

            self.heap_link(node, neighbour)
            root_list[degree] = None
            degree += 1

        root_list[degree] = node

        if node == self.min_node:
            break

        node = node.right

    for node in root_list:
        if node is not None:
            if node.key < self.min_node.key:
                self.min_node = node
```

`heap_link` will create a parent-child link between two nodes. It means that after having consolidated the heap, no nodes of same degree can be in the root list. 

After removing `min_node`,  its children will become roots of new trees. If the number of children was d, it takes time **$O(d)$** to process all new roots. Therefore, the amortized running time of this phase is **$O(d) = O(log(n))$**.

To complete the `extract_min` operation, we need to update the pointer to the root with minimum key. Unfortunately there may be up to n roots we need to check. In the second phase we therefore decrease the number of roots by successively linking together roots of the same degree. When two roots u and v have the same degree, we make one of them a child of the other so that the one with the smaller key remains the root. Its degree will increase by one. This is repeated until every root has a different degree. To find trees of the same degree efficiently we use an array of length O(log n) in which we keep a pointer to one root of each degree. When a second root is found of the same degree, the two are linked and the array is updated. The actual running time is O(log n + m) where m is the number of roots at the beginning of the second phase. At the end we will have at most O(log n) roots (because each has a different degree). Therefore, the difference in the potential function from before this phase to after it is: O(log n) − m, and the amortized running time is then at most O(log n + m) + c(O(log n) − m). With a sufficiently large choice of c, this simplifies to O(log n).

In the third phase we check each of the remaining roots and find the minimum. This takes O(log n) time and the potential does not change. The overall amortized running time of extract minimum is therefore O(log n).



### Graph

The `graph.py` file is composed of a main class `Graph` and its 3 subclasses `GraphNode`, `GraphEdge`, `GraphRoute`. Furthermore, some functions are defined outside classes either being requested in class methods, participating to the construction of the graph or allowing the graph to be displayed. 

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

This adaptation of *Dijkstra's algorithm* uses a priority queue data structure which takes form as a *Fibonacci Heap*. This allows the decrease_key method to run in **$O(1)$** time complexity and the extract_min method in **$O(log(n))$** time complexity (where *n* represents the number of nodes in the heap). A full implementation is available in the `heap.py` file and will be discuss further in the according section.

Therefore, this algorithm runs in **$O(|E| + |V|log|V|)$** (where *|E|* represents the number of edges and *|V|* the number of vertices) time complexity. In the method, `distances`, `parents` and `heap` variables have a **$O(|V|)$** memory complexity which defines that of `get_path_given_power`.

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

*Kruskal's algorithm* can be shown to run in **$O(|E|log|V|)$** time complexity,  all with simple data structures. However, the `graph_to_tree` call will add another **$O(|V|)$** time complexity as it browses every vertices from the new graph to create the tree. Hence, the overall time complexity is **$O(|V|+|E|log|V|)$**. 

Time complexity comparison between *Kruskal's algorithm* and *Dijkstra's algorithm* adaptations: 

**$$ |V|+|E|log(|V|)-(|E|+|V|log(|V|) = (|V|-|E|)(1-log(|V|) \geq 0   $$**

*Kruskal's algorithm* has a greater time complexity than *Dijkstra's algorithm* however both alogrithm have not the same purpose and the first one can save a lot amount of time for the goal of this project. Indeed, each route will have to be updated with the minimum power to cover them. By using *Dijkstra's algorithm* adaptation, each call will make **$O(|E| + |V|log|V|)$** time, therefore the total time of this update will be **$O(|R||E| + |R||V|log|V|))$** (where *|R|* is the number of routes). On the other hand, *Kruskal's algorithm* adaptation will be run only one time and then the time complexity of updating a route is **$O(|V|)$** (a detailed explanation is made on the *Tree* paragraph). Therefore, the total cost of the update is **$O(|V|+|E|log|V|+|R||V|)$**. The overall time complexity is better in this case. While running the simulation, this improvement of efficiency can be observed as the estimation of updating *network.2.in* is around 550 hours and *network.3.in* takes to long to estimate. On the other hand, the estimations for the different networks using *Kruskal's algorithm* adaptation range from 0 second to 76 seconds.

### Tree

This is a well known data structure in python. The `tree.py` file is composed of a main class `Tree` and its subclass `TreeNode`. Some functions are defined outside classes being requested in class methods.  

A representation of the tree goes as follow (the network station is the root of the tree):

<img src="https://user-images.githubusercontent.com/118286479/230014823-cb4ccced-80b1-4272-90a4-e054db30d53f.png" width="300" height="400">

***Attributes description***

- `TreeNode` represents a node in the tree. It contains the node `value` representing the value of a unique vertex in the network. Furthermore, the tree data structure requires that each node of the tree can have `children` and a `parent`. A `degree` attribute keeps track of the node depth in the tree which is really useful to have a better efficiency in the methods presented below. A particularity of this implementation is the presence of attributes `power` and `distance`. Indeed, parents and children are neighbours in the network therefore the link between a tree node and its parent is actually an edge in the graph with its own power and distance. Those attributes are here to represent the characteristics of the edge (*in the tree represented above, **2** has **7** and **5** as neighbours, so the distance of the edge **2 -- 7** is stored in `distance` attribute of **7**)*. To add a more realistic constraint in the simulation, an attribute `broke` is set to represent the possibility that an edge in the network is bloked and the route can no longer be covered. 

- `Tree` keeps track of the network `gas_price` and associate each node to its value in `nodes`. Moreover it contains the `root`of the tree. 

***Methods description***

The `Tree` class contains one main method `route_characteristics` which finds the route characteristics between two nodes using their lowest common ancestor (*LCA*). The method uses three other functions and methods that will be explained further in the following segment. `lowest_common_ancestor` is basically finding the *LCA* of two nodes. When two nodes share their *LCA*, `characteristics_until_lca` is calculating the path, distance and power from a node to its *LCA*. Therefore the algorithm has to iterate through each parent nodes. The `iterate_from_node_to_lca` method is doing such work as it is basically a generator. 

```ruby
def lowest_common_ancestor(node1, node2):
    while node1.degree > node2.degree:
        node1 = node1.parent

    while node2.degree > node1.degree:
        node2 = node2.parent

    while node1 != node2:
        node1 = node1.parent
        node2 = node2.parent

    return node1
```

By keeping track of the nodes degree, the `lowest_common_ancestors* method runs efficiently because it iterates only through the parent nodes until the *LCA*.  

```ruby
def characteristics_until_lca(self, node, lca, broke=True):
  path = []
  power = 0
  distance = 0
  route_available = True
  nodes = self.iterate_from_node_to_lca(node, lca)
  for node in nodes:
      if node.broke and broke:
          route_available = False

      if node != lca:
          power = max(power, node.power)
          distance += node.distance
          path.append(node.value)

  return path, power, distance, route_available
```

`characteristics_until_lca` is interesting in such a way that it uses a generator `iterate_from_node_to_lca` enabling the possibility to not store every data beforhand and reaching a better memory complexity. Such technique could be used in other methods to optimize their memory complexity.   

```ruby
def iterate_from_node_to_lca(self, start_node, lca):
    node = start_node

    while node != lca:    
        yield node
        node = node.parent

    yield lca
```

From the methods shown above, the overall time complexity of getting route characteristics (including the minimum power) is **$O(|V|)$** from the minimum spanning tree of the graph. 


### Delivery Network

`delivery_network.py` will not be fully explained. However the purpose of creating such file was to get the best allocation of trucks (given in the truck.i.in files) with a given method. To find such allocation, an adaptation of a *genetic algorithm* is being used as it allows to find a good option quickly. The algorithm runs during a given time. 20 seconds is enough time to get a satisfying option for all networks.  


