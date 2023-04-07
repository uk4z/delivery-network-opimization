import sys 
sys.path.append("D:\Coding files\Delivery network")

import unittest 
from test import graph_from_file

class Test_GraphLoading(unittest.TestCase):
    def test_network0(self):
        graph_nodes, graph_edges = graph_from_file("input/network.00.in")
        self.assertEqual(len(graph_nodes), 10)
        self.assertEqual(len(graph_edges), 9)

    def test_network1(self):
        graph_nodes, graph_edges = graph_from_file("input/network.01.in")
        self.assertEqual(len(graph_nodes), 7)
        self.assertEqual(len(graph_edges), 5)
    
    def test_network4(self):
        graph_nodes, graph_edges = graph_from_file("input/network.04.in")
        self.assertEqual(len(graph_nodes), 10)
        self.assertEqual(len(graph_edges), 4)

if __name__ == '__main__':
    unittest.main()