class Graph:
    def __init__(self, nodes=[]):
        """
        Initializes a graph with an optional list of nodes.
        Parameters:
        -----------
        nodes: list, optional
            A list of nodes to initialize the graph with. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    
    def __str__(self):
        """Prints the graph as a list of neighbors for each 
        node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes \
            and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, 
        hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        # Adds new nodes in the list
        if node1 not in self.nodes:
            self.nodes.append(node1)
            self.nb_nodes += 1
        if node2 not in self.nodes: 
            self.nodes.append(node2)
            self.nb_nodes += 1
        # Adds edge to the graph
        self.graph[node1] = self.graph.get(node1, []) +\
            [[node2, power_min, dist]]
        self.graph[node2] = self.graph.get(node2, []) +\
            [[node1, power_min, dist]]
        # An edge has been created
        self.nb_edges += 1

    def get_path_with_power(self, src, dest, power):
        """
        The algorithm goes through every nodes and for each node, 
        it browses every edges thus the complexity is in O(V.E) 
        where V and E represent respectively the number of nodes and edges.
        Only takes road with the minimal distance 

        Args:
            src (int): the source of the road
            dest (int): the destination of the road
            power (int): the power of the truck

        Returns:
            list: Contains the different nodes which compose the road 
        """
        res = [[]]
        visit = []
        self.minimal_length = float('inf')

        def road(node, curr, length):
            if node == dest:
                if length < self.minimal_length:
                    self.minimal_length = length
                    res[0] = curr
                return
            if node in visit: 
                return 
            else: 
                visit.append(node)
                for neighbor in self.graph[node]:
                    if neighbor[1] <= power:
                        road(neighbor[0], curr+[neighbor[0]], length + neighbor[2])
        road(src, [src], 0)
        return res[0] if res[0] else None
    
    def connected_components(self):
        """
        The algorithm goes through every node of the graph. 
        For each node, it browses each neighbor of the considered node. 
        Therefore the complexity is in O(V.E) where V and E represent 
        respectively the number of nodes and edges .
        Returns:
            dictionary: contains all connections between nodes
        """
        connection = {}
        for key, val in self.graph.items():
            for i in range(len(val)):
                connection[key] = connection.get(key, []) + [val[i][0]]
        return connection 
    
    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: 
        {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        result = set()

        def road(node, curr):
                if node in visit: 
                    return 
                else: 
                    visit.add(node)
                    for neighbor in self.graph[node]:
                         road(neighbor[0], curr+[neighbor[0]])
        
        for node in self.graph.keys():
            if node not in result:
                visit = set()
                road(node, [node])
                result.add(frozenset(visit))
                
        return result 
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        The algorithm has the same complexity as get_path_with_power
        """
        res = [[], 0]
        self.minimal_power = float('inf')

        def road(node, curr, power):
            if node == dest:
                if self.minimal_power > power:
                    self.minimal_power = power
                    res[0] = curr
                    res[1] = power
                return
            if node in curr[:len(curr)-1]: 
                return 
            else: 
                for neighbor in self.graph[node]:
                    temp = max(power, neighbor[1])
                    road(neighbor[0], curr+[neighbor[0]], temp)
        road(src, [src], 0)
        return res if res else None


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 
        power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    G: Graph
        An object of the class Graph with the graph from file_name.
    """
    # Opens the file
    with open(filename, 'r') as f:
        text = f.read()
        # Selects lines 
        L = text.split('\n')
        # Creates the graph according to the correct format
        nb_nodes = L[0].split(' ')[0]
        nodes = [i+1 for i in range(int(nb_nodes))]
        G = Graph(nodes)
        # Builds edges between nodes 
        for i in range(1, len(L)):
            values = L[i].split(' ')
            if len(values) == 4: #checks if a distance is mentionned
                node1, node2, power_min, dist = values
                G.add_edge(int(node1), int(node2), int(power_min), int(dist))
            else: #the distance is not mentionned
                node1, node2, power_min = values
                G.add_edge(int(node1), int(node2), int(power_min))
        return G


def plot_graph(g):
    from graphviz import Source

    temp = """ graph{ """ 

    visit = []
    for key, val in g.graph.items():
        visit.append(key)
        for neighbor in val:
            if neighbor[0] not in visit:
                temp += str(key) + "--" + str(neighbor[0]) + """[label= "p = """ + str(neighbor[1]) + ";d = " + str(neighbor[2]) + """ "]""" + ";" + "\n"
    temp += """}"""

    s = Source(temp, filename="Graph.gv", format="png")
    s.view()
