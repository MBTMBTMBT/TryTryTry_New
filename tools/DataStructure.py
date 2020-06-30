class Node(object):

    def __init__(self, element):
        self.element = element

    def __str__(self):
        try:
            return str(self.element)
        except Exception:
            pass


class LinkedListNode(Node):

    def __init__(self, element, belongs_to=None, previous=None, next=None):
        super(LinkedListNode, self).__init__(element)
        self.belongs_to = belongs_to
        self.next = next
        self.previous = previous


class LinkedList(object):

    def __init__(self, name=None):
        self.first = None
        self.last = None
        self.size = 0
        self.name = name

    def __len__(self):
        return self.size

    def is_empty(self) -> bool:
        return self.size == 0

    def list(self) -> []:
        rst = []
        present = self.first
        while present is not None and present is not self.last:
            rst.append(present)
            present = present.next
        return rst

    def add_node(self, node: LinkedListNode) -> LinkedListNode:
        if self.is_empty():
            self.first = node
            self.last = node
        else:
            self.last.next = node
            node.previous = self.last
            node.next = None
            self.last = node
        self.size += 1
        node.belongs_to = self
        return node

    def add_element(self, element) -> LinkedListNode:
        node = LinkedListNode(element)
        return self.add_node(node)

    def remove(self, node: LinkedListNode) -> LinkedListNode:
        if node.belongs_to is not self:
            raise RuntimeWarning("Node does not belong to this List!")
        if self.size == 1:
            self.first = None
            self.last = None
        elif node is self.first:
            self.first = node.next
        elif node is self.last:
            self.last = node.previous
        else:
            node.previous.next = node.next
            node.next = node.previous.next
        node.next = None
        node.previous = None
        node.belongs_to = None
        self.size -= 1
        return node

