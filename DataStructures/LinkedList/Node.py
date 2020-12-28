class Node:
    def __init__(self, data, n=None):
        self.data = data
        self.next = n


class LinkedList:
    def __init__(self, r=None):
        self.size = 0
        self.root = r

    def get_size(self):
        return self.size

    def add(self, d):
        self.root = Node(d, self.root)
        self.size += 1

    def remove(self, d):
        this_node = self.root
        prev_node = None
        while this_node:
            if this_node.data() == d:
                if prev_node:
                    # connect previous node to the next node
                    prev_node.set_next(this_node.next())
                else:
                    self.root = this_node.next
                self.size -= 1
                return True
            else:
                prev_node = this_node
                this_node = prev_node.next()

    def remove_head(self):
        this_node = self.root

        self.root = this_node.next
        return self.root

    def find(self, d):
        this_node = self.root
        while this_node:
            if this_node.data == d:
                return True
            else:
                this_node = this_node.next()
        return False

    def reversed(self):
        this_node = self.root
        prev = None
        while this_node:
            curr = this_node
            this_node = this_node.next
            curr.next = prev
            prev = curr
        return prev


if __name__ == '__main__':
    this_list = LinkedList()
    for i in range(5):
        this_list.add(i)
    head = this_list.root
    while head:
        print(head.data)
        head = head.next
