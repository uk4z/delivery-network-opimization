import sys 
sys.path.append("D:\Coding files\Delivery network")

import unittest 
from graph import graph_from_file

class Test_GraphLoading(unittest.TestCase):
    def test_network0(self):
        g = graph_from_file("input/network.00.in")
        k = kruskal(g)
        self.assertEqual(k.nb_nodes, 10)
        self.assertEqual(k.nb_edges, 9)

    def test_network1(self):
        g = graph_from_file("input/network.01.in")
        k = kruskal(g)
        self.assertEqual(k.nb_nodes, 7)
        self.assertEqual(k.nb_edges, 5)
    
    def test_network2(self):
        g = graph_from_file("input/network.02.in")
        k = kruskal(g)
        self.assertEqual(k.nb_nodes, 10)
        self.assertEqual(k.nb_edges, 9)

    def test_network3(self):
        g = graph_from_file("input/network.03.in")
        k = kruskal(g)
        self.assertEqual(k.nb_nodes, 10)
        self.assertEqual(k.nb_edges, 9)

    def test_network4(self):
        g = graph_from_file("input/network.04.in")
        k = kruskal(g)
        self.assertEqual(k.nb_nodes, 10)
        self.assertEqual(k.nb_edges, 9)

if __name__ == '__main__':
    unittest.main()