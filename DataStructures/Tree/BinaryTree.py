from collections import deque


class Node:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def insert_into_bst(self, root, val):
        node = Node(val=val)
        if not root:
            return node
        else:
            if val > root.val:
                root.right = self.insert_into_bst(root.right, val)
            if val < root.val:
                root.left = self.insert_into_bst(root.left, val)
        return root

    def display_tree(self):
        if self.left:
            self.left.display_tree()
        print(self.val),
        if self.right:
            self.right.display_tree()


class InOrderTraversal:
    def in_order_traversal(self, root):
        res = []
        self.in_order_recursion(root, res)
        return res

    def in_order_recursion(self, node, res):
        if not node:
            return res
        else:
            self.in_order_recursion(node.left, res)
            res.append(node.val)
            self.in_order_recursion(node.right, res)


class BFS:
    def bfs(self, root):
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            size = len(queue)
            for index in range(size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                level.append(node.val)
            res.append(level)
        return res


if __name__ == '__main__':
    root = Node(30)
    for i in [40,10,50,20,5,35]:
        root = root.insert_into_bst(root, i)
    root.display_tree()
    inorder = InOrderTraversal()
    print(inorder.in_order_traversal(root))

    bfs = BFS()
    print(bfs.bfs(root))

