import sys 
sys.path.append("D:\Coding files\Delivery network")

from test import graph_from_file, connected_components_set

import unittest   # The test framework

class Test_GraphCC(unittest.TestCase):
    def test_network0(self):
        graph_nodes = graph_from_file("input/network.00.in")[0]
        result = connected_components_set(graph_nodes)
        self.assertEqual(result, {frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10})})

    def test_network1(self):
        graph_nodes = graph_from_file("input/network.01.in")[0]
        result = connected_components_set(graph_nodes)
        self.assertEqual(result, {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})})

if __name__ == '__main__':
    unittest.main()