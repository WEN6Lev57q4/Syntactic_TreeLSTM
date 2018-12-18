from queue import Queue
import numpy as np
class Tree(object):
    def __init__(self, index):
        self.root = None
        self.node_array = {}
        node = TreeNode(index, None)
        node.set_word('(')
        self.root = node
        self.node_array[0] = node

    def add_children(self, parent_index, index, word):
        node = TreeNode(index, parent_index)
        node.set_word(word)
        self.node_array[index] = node
        self.node_array[parent_index].add_children(node)

    def resort_sequence(self):
        new_array = []
        tree_to_bfs_to_reverse = []
        reverse_to_bfs_to_tree = {}
        q = Queue()
        q.put(self.root)
        index = 0
        while not q.empty():
            node = q.get()
            new_array.append(node)
            tree_to_bfs_to_reverse.append(node.index)
            index += 1
            children = node.children
            children.reverse()
            for child in children:
                q.put(child)

        new_array.reverse()
        tree_to_bfs_to_reverse.reverse()

        for reverse, tree in enumerate(tree_to_bfs_to_reverse):
            reverse_to_bfs_to_tree[tree] = reverse

        self.new_array = new_array
        self.reverse_to_bfs_to_tree = reverse_to_bfs_to_tree
        return

    def return_sequence(self):
        sequence = ' '.join([node.get_word() for node in self.new_array])
        return sequence

    def convert_to_graph(self):
        node_num = len(self.node_array.keys())
        graph = np.zeros((node_num, node_num), dtype='float32')
        for parent_index, node in enumerate(self.new_array):
            for child in node.children:
                child_index = self.reverse_to_bfs_to_tree[child.index]
                graph[parent_index][child_index] = 1
        return graph


def print_tree(tree):

    if not tree.has_children():
        print(tree.get_word())
    else:
        for child in tree.children:
            print_tree(child)
        print(tree.get_word())
    return 0


class TreeNode(object):
    def __init__(self,index,parent):
        self.index = index
        self.parent = parent
        self.children = []
        self.num_children = 0
        self.word = None

    def add_children(self,Node):
        self.children.append(Node)
        self.num_children+=1

    def set_word(self,word):
        self.word = word

    def get_word(self):
        return self.word

    def has_children(self):
        return not len(self.children)== 0


if __name__ == '__main__':
    pass
