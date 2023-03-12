import  unittest
from heap import FibonacciHeap


def heap_from_file(filename):
    heap = FibonacciHeap()
    lines = open_heap_file(filename)
    for line in lines:
        node = get_data_from_line(line)
        value, key = node 
        heap.insert_Node(value, key)
    return heap 

def open_heap_file(filename):
    with open(filename) as file:
        input = file.read()
        lines = input.splitlines()
        return lines 
    
def get_data_from_line(line):
    raw_data = line.split(' ')
    processed_data = [int(data) for data in raw_data]
    return processed_data

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
        child = FibonacciHeap.Node(2,11)
        node.new_child(child)
        self.assertTrue(node.have_child)
    
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

    def test_kill_children(self):
        node = FibonacciHeap.Node(1, 10)
        child1 = FibonacciHeap.Node(2, 11)
        child2 = FibonacciHeap.Node(3,13)
        node.new_child(child1)
        node.new_child(child2)
        node.kill_children()
        self.assertIsNone(child1.parent)
        self.assertIsNone(child2.parent)

    def test_get_min_child(self):
        node = FibonacciHeap.Node(1, 10)
        self.assertIsNone(node.get_min_child())
        child1 = FibonacciHeap.Node(2, 11)
        child2 = FibonacciHeap.Node(3,13)
        node.new_child(child1)
        node.new_child(child2)
        min_child = node.get_min_child()
        self.assertEqual(min_child.value, 2)

    def test_get_min_neighbor(self):
        node = FibonacciHeap.Node(1, 10)
        self.assertIsNone(node.get_min_neighbor())
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
        node.new_neighbor(neighbor1)
        neighbor2 = FibonacciHeap.Node(3,10)
        node.new_neighbor(neighbor2)
        node.kill_neighbors()
        self.assertEqual(neighbor1.right, neighbor2)

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
        self.assertFalse(root1.same_depth(child2))

    def test_map(self):
        node1 = FibonacciHeap.Node(1, 10)
        node3 = FibonacciHeap.Node(3, 11)
        node4 = FibonacciHeap.Node(4, 12)
        node1.new_child(node3)
        node1.new_child(node4)
        node2 = FibonacciHeap.Node(2, 12)
        node5 = FibonacciHeap.Node(5, 13)
        node6 = FibonacciHeap.Node(6, 8)
        node2.new_child(node5)
        node1.new_neighbor(node2) 
        node1.new_neighbor(node6) 
        node1.map()
        self.assertEqual(node1.child.right.value, 2)
        self.assertEqual(node1.right, node6)

    def test_change_child_from_children(self):
        node = FibonacciHeap.Node(1,10)
        child = FibonacciHeap.Node(2,11)
        node.new_child(child)
        node.change_child_from_children()
        self.assertEqual(node.child.value, 2)
        new_child = FibonacciHeap.Node(3,12)
        node.new_child(new_child)
        node.change_child_from_children()
        self.assertEqual(node.child.value, 3)

    def test_neighbors(self):
        node = FibonacciHeap.Node(1,10)
        neighbor1 = FibonacciHeap.Node(2,4)
        neighbor2 = FibonacciHeap.Node(3,5)
        node.new_neighbor(neighbor1)
        node.new_neighbor(neighbor2)
        neighbors = node.neighbors()
        self.assertIn(neighbor1, neighbors)
        self.assertIn(neighbor2, neighbors)
        self.assertIn(node, neighbors)

class Test_FibonacciHeap(unittest.TestCase):
    def test_is_empty(self):
        heap = FibonacciHeap()
        self.assertTrue(heap.is_Empty())
        heap.insert_Node(1,1)
        self.assertFalse(heap.is_Empty())

    def test_insert_Node(self):
        heap = FibonacciHeap()
        heap.insert_Node(1,10)
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
        heap1 = heap_from_file("Fibonacci Heap/heap.00.in")
        heap2 = heap_from_file("Fibonacci Heap/heap.01.in")
        heap1.union(heap2)
        self.assertEqual(heap1.min_node.value, heap2.min_node.value, 1)

    def test_heap_reorganisation(self):
        heap1 = heap_from_file("Fibonacci Heap/heap.00.in")
        heap2 = heap_from_file("Fibonacci Heap/heap.01.in")
        heap3 = heap_from_file("Fibonacci Heap/heap.02.in")
        heap1.union(heap2)
        heap1.union(heap3)
        heap1.heap_reorganisation()
        self.assertEqual(heap1.min_node.right.child.value, 5)
        self.assertEqual(heap1.min_node.child.right.child.right.value, 8)

    def test_cut(self):
        heap = heap_from_file("Fibonacci Heap/heap.00.in")
        parent = heap.min_node
        heap.cut(parent.child, parent)
        self.assertEqual(parent.right.value, 2)
        self.assertEqual(parent.child.value, 3)

    def test_decrease_key(self):
        heap = heap_from_file("Fibonacci Heap/heap.00.in")
        parent = heap.min_node
        heap.decrease_key(parent.child, 8)
        self.assertEqual(parent.right.value, 2)
        self.assertEqual(parent.child.value, 3)

    def test_extract_min(self):
        heap = heap_from_file("Fibonacci Heap/heap.00.in")
        value = heap.extract_min()
        self.assertEqual(value, 1)
        self.assertEqual(heap.min_node.value, 2)

    def test_have_node(self):
        heap = heap_from_file("Fibonacci Heap/heap.00.in")
        heap2 = heap_from_file("Fibonacci Heap/heap.01.in")
        heap3 = heap_from_file("Fibonacci Heap/heap.02.in")
        heap.union(heap2)
        heap.union(heap3)
        node = heap.min_node.right.child.right
        self.assertTrue(heap.have_node(node))

    def test_get_node_in_heap(self):
        heap = heap_from_file("Fibonacci Heap/heap.00.in")
        heap2 = heap_from_file("Fibonacci Heap/heap.01.in")
        heap3 = heap_from_file("Fibonacci Heap/heap.02.in")
        heap.union(heap2)
        heap.union(heap3)
        node = heap.get_node_in_heap(6)
        self.assertEqual(heap.min_node.right, node )

if __name__ == "__main__":
    unittest.main()

