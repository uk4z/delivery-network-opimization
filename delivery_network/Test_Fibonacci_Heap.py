import  unittest
from Fibonacci_Tree import FibonacciHeap, FibonacciHeapNode

class Test_FibonacciHeapNode(unittest.TestCase):
    def test_link_to_the_left(self):
        node1 = FibonacciHeapNode(1, 1)
        node2 = FibonacciHeapNode(2, 2)
        node1.link_to_the_left(node2)
        self.assertEqual(node1.right, node2, node2.left.left)
        node3 = FibonacciHeapNode(3, 3)
        node3.link_to_the_left(node2)
        self.assertEqual(node1.right.right.right, node1)
        self.assertEqual(node3.left, node1)

    def test_suppress_neighbors(self):
        node1 = FibonacciHeapNode(1, 1)
        node2 = FibonacciHeapNode(2, 2)
        node3 = FibonacciHeapNode(3, 3)
        node1.link_to_the_left(node2)
        node3.link_to_the_left(node2)
        node1.suppress_neighbors()
        self.assertEqual(node1, node1.right)
        self.assertEqual(node2.right, node3)

    def test_link_as_child(self):
        node1 = FibonacciHeapNode(1, 1)
        node2 = FibonacciHeapNode(2, 2)
        node3 = FibonacciHeapNode(3, 3)
        node2.link_as_child(node1)
        node3. link_as_child(node1)
        self.assertEqual(node2, node1.child.right.right)
        self.assertEqual(node3.parent, node1)
    

class Test_FibonacciHeap(unittest.TestCase):

    def test_insertion(self):
        heap = FibonacciHeap()
        nodes = []
        for i in range(10, 0, -1):
            heap.insertion(i, i + 1)
            self.assertEqual(heap.min_node.wrap, i)
            nodes.append(heap.min_node)
        # nodes = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        self.assertEqual(heap.nb_nodes, 10)
        self.assertEqual(nodes[0].right, nodes[-1])

    def test_have_wrap(self):
        heap = FibonacciHeap()
        for i in range(1, 11):
            heap.insertion(i, i + 1)
        self.assertTrue(heap.have_wrap(3))
        self.assertTrue(heap.have_wrap(10))
        self.assertFalse(heap.have_wrap(24))

    def test__get_node_by_wrap(self):
        heap = FibonacciHeap()
        nodes = []
        for i in range(1, 11):
            heap.insertion(i, i + 1)
            nodes.append(heap.min_node.left)
        # nodes = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        self.assertEqual(heap._get_node_by_wrap(10), nodes[-1])
        self.assertIsNone(heap._get_node_by_wrap(234))

    def test_consolidate(self):
        heap = FibonacciHeap()
        nodes = []
        for i in range(1, 11):
            heap.insertion(i, i + 1)
            nodes.append(heap.min_node.left)
        # nodes = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        heap.consolidate()
        self.assertEqual(nodes[-2], heap.min_node.right)
        self.assertEqual(nodes[-3], heap.min_node.child.left.child.left.child)

    def test_extract_min(self):
        heap = FibonacciHeap()
        nodes = []
        for i in range(1, 11):
            heap.insertion(i, i + 1)
            nodes.append(heap.min_node.left)
        # nodes = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        output = heap.extract_min()
        self.assertEqual(output, 1)
        self.assertEqual(nodes[2], heap.min_node.child)
        self.assertEqual(nodes[4], heap.min_node.child.right.child)

    def test_decrease_key(self):
        heap = FibonacciHeap()
        nodes = []
        for i in range(1, 11):
            heap.insertion(i, i + 1)
            nodes.append(heap.min_node.left)
        # nodes = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        heap.extract_min()
        heap.decrease_key(9, 1)
        self.assertEqual(heap.min_node, nodes[-2])
        self.assertEqual(heap.min_node.right, nodes[1])



if __name__ == "__main__":
    unittest.main()
