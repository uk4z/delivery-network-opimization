import  unittest
from heap import FibonacciHeap



class Test_FibonacciHeap_Node(unittest.TestCase):
    def test_have_neighbors(self):
        node = FibonacciHeap.Node(1,10)
        self.assertFalse(node.have_neighbors())
        neighbor = FibonacciHeap.Node(2,10)
        node.set_neighbor(neighbor)
        self.assertTrue(node.have_neighbors())

    def test_set_neighbor(self):
        node = FibonacciHeap.Node(1,10)
        neighbor = FibonacciHeap.Node(2,10)
        node.set_neighbor(neighbor)
        self.assertEqual(node.right.value, 2)

    def test_add_neighbor(self):
        node = FibonacciHeap.Node(1,10)
        neighbor = FibonacciHeap.Node(2,10)
        node.set_neighbor(neighbor)
        new_neighbor = FibonacciHeap.Node(3,10)
        node.add_neighbor(new_neighbor)
        self.assertEqual(node.right.value, 3)
    
    def test_new_neighbor(self):
        node = FibonacciHeap.Node(1,10)
        neighbor1 = FibonacciHeap.Node(2,10)
        neighbor2 = FibonacciHeap.Node(3,10)
        node.new_neighbor(neighbor1)
        self.assertEqual(node.right.value, 2)
        node.new_neighbor(neighbor2)
        self.assertEqual(node.right.value, 3)
    
    def test_have_child(self):
        node = FibonacciHeap.Node(1,10)
        self.assertFalse(node.have_child())
    
    def test_set_child(self):
        node = FibonacciHeap.Node(1,10)
        child = FibonacciHeap.Node(2,11)
        node.set_child(child)
        self.assertEqual(node.child.value, 2)
        self.assertEqual(child.parent.value, 1)

    def test_add_child(self):
        node = FibonacciHeap.Node(1,10)
        child = FibonacciHeap.Node(2,11)
        node.set_child(child)
        new_child = FibonacciHeap.Node(3,11)
        node.add_child(new_child)
        self.assertEqual(node.child.right.value, 3)
        self.assertEqual(new_child.parent.value, 1)

    def test_new_child(self):
        node = FibonacciHeap.Node(1,10)
        child1 = FibonacciHeap.Node(2,11)
        child2 = FibonacciHeap.Node(3,11)
        node.new_child(child1)
        self.assertEqual(node.child.value, 2)
        node.new_child(child2)
        self.assertEqual(node.child.right.value, 3)

    def test_separation_children(self):
        node = FibonacciHeap.Node(1, 10)
        child1 = FibonacciHeap.Node(2, 11)
        child2 = FibonacciHeap.Node(3,13)
        node.new_child(child1)
        node.new_child(child2)
        min_node = node.separation_children()
        self.assertEqual(min_node.value, 2)

    def test_get_min_neighbor(self):
        node = FibonacciHeap.Node(1, 10)
        min_neighbor = node.get_min_neighbor()
        self.assertEqual(min_neighbor.value, 1)
        neighbor1 = FibonacciHeap.Node(2,12)
        neighbor2 = FibonacciHeap.Node(3,4)
        node.new_neighbor(neighbor1)
        node.new_neighbor(neighbor2)
        min_neighbor = node.get_min_neighbor()
        self.assertEqual(min_neighbor.value, 3)

    def test_kill_neighbors(self):
        node = FibonacciHeap.Node(1,10)
        neighbor1 = FibonacciHeap.Node(2,10)
        node.new_neighbor(neighbor1)
        node.kill_neighbors()
        self.assertEqual(neighbor1.right, neighbor1)
        neighbor1.new_neighbor(node)
        neighbor2 = FibonacciHeap.Node(3,10)
        node.add_neighbor(neighbor2)
        node.kill_neighbors()
        self.assertEqual(neighbor1.right, neighbor2)

    def test_kill_children(self):
        node = FibonacciHeap.Node(1,10)
        child = FibonacciHeap.Node(2,11)
        node.new_child(child)
        node.kill_children()
        self.assertIsNone(node.child)

    def test_get_depth(self):
        node = FibonacciHeap.Node(1,10)
        child1 = FibonacciHeap.Node(2,11)
        child2 = FibonacciHeap.Node(3,11)
        node.new_child(child1)
        node.new_child(child2)
        node_depth = node.get_depth() 
        self.assertEqual(node_depth, 1)   
        new_node = FibonacciHeap.Node(4,13)
        child1.new_child(new_node)
        node_depth = node.get_depth()
        self.assertEqual(node_depth, 2)          

    def test_same_depth(self):
        root1 = FibonacciHeap.Node(1,10)
        child1 = FibonacciHeap.Node(2,11)
        child_neighbor = FibonacciHeap.Node(5,12)
        root1.new_child(child1)
        root1.new_child(child_neighbor)
        root2 = FibonacciHeap.Node(3,10)
        child2 = FibonacciHeap.Node(4,11)
        root2.new_child(child2)
        self.assertTrue(root1.same_depth(root2))

    def test_map(self):
        node1 = FibonacciHeap.Node(1, 10)
        node3 = FibonacciHeap.Node(3, 11)
        node4 = FibonacciHeap.Node(4, 12)
        node1.new_child(node3)
        node1.new_child(node4)
        node2 = FibonacciHeap.Node(2, 12)
        node5 = FibonacciHeap.Node(5, 13)
        node2.new_child(node5)
        node1.new_neighbor(node2)  
        node1.map()
        self.assertEqual(node1.child.right.value, 2)

class Test_FibonacciHeap(unittest.TestCase):
    def test_is_empty(self):
        heap = FibonacciHeap()
        self.assertTrue(heap.is_Empty())
        heap.insert_Node(1,1)
        self.assertFalse(heap.is_Empty())

    def test_insert_Node(self):
        heap = FibonacciHeap()
        heap.insert_Node(1,10)
        self.assertEqual(heap.nb_nodes, 1)
        heap.insert_Node(2, 8)
        self.assertEqual(heap.min_node.value, 2 )
        heap.insert_Node(3,8)
        self.assertEqual(heap.min_node.right.value, 3)
        heap.insert_Node(4,9)
        self.assertEqual(heap.min_node.child.right.value, 4)

    def test_get_min(self):
        heap = FibonacciHeap()
        self.assertIsNone(heap.get_min())
        heap.insert_Node(1,10)
        min_node = heap.get_min()
        self.assertEqual(min_node.value, 1)

    def test_union(self):
        heap1 = FibonacciHeap()
        heap1.insert_Node(1,10)
        heap2 = FibonacciHeap()
        heap2.insert_Node(2,9)
        heap1.union(heap2)
        self.assertEqual(heap1.min_node.value, heap2.min_node.value, 2)

    def test_extract_min(self):
        heap = FibonacciHeap()
        heap.insert_Node(1,10)
        heap.insert_Node(2,10)
        heap.insert_Node(3,11)
        value = heap.extract_min()
        self.assertEqual(value, 1)
        self.assertEqual(heap.min_node.value, 2)

    def test_heap_reorganisation(self):
        heap1 = FibonacciHeap()
        heap2 = FibonacciHeap()
        heap3 = FibonacciHeap()
        heap1.insert_Node(1,10)
        heap1.insert_Node(2,12)
        heap1.insert_Node(3,13)
        heap2.insert_Node(4,11)
        heap2.insert_Node(5,12)
        heap3.insert_Node(6,11)
        heap3.insert_Node(7,12)
        heap3.insert_Node(8,13)
        heap1.union(heap2)
        heap1.union(heap3)
        heap1.heap_reorganisation()
        self.assertEqual(heap1.min_node.right.child.value, 5)

if __name__ == "__main__":
    unittest.main()

