class LRUNode:
    def __init__(self, value, next=None, prev=None):
        self.value = value
        self.next = next
        self.prev = prev


class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.head = LRUNode(None, None)
        self.tail = LRUNode(None, None)
        self.dct = {}
        self.size = 0
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        if key in list(self.dct.keys()):
            node = self.dct[key]
            self.move_to_front(key)
            return node.value
        else:
            return None

    def put(self, key, value):
        if key in self.dct.keys():
            node = self.dct[key]
            node.value = value
            self.move_to_front(key)
        else:
            if len(self.dct) >= self.capacity:
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            newnode = LRUNode(key)
            newnode.prev = self.tail.prev
            self.tail.prev = newnode
            newnode.next = self.tail
            self.dct[key] = newnode

    def move_to_front(self, key):
        node = self.dct[key]
        node.prev = self.tail.prev
        self.tail.prev = node
        node.next = self.tail
        self.dct[key] = node


if __name__ == '__main__':
    lru = LRU(5)
    lru.put(1, 1)
    lru.put(2, 2)
    print(lru.get(2))
    print(lru.dct)