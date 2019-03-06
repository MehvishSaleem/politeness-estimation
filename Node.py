# Node class for one node in the Tree
import numpy as np


class Node():

    def __init__(self, vec=None):
        self.vec = vec
        self.word = ''
        self.children = []
        self.num_child = 0

    def add_child(self, node):
        """
        Adds a child to this node of the tree
        """
        if self.num_child < 2:
            self.children.append(node)
            self.num_child += 1

    def read(self, line, index, init):
        """
        Reads a complete line from the file.
        Each line contains one binary tree in the PTB format.
        Constructs the node, takes as input its word, and calls the function recursively on its children.
        """
        num_children = 0
        word = ''
        i = index

        line_len = len(line)
        while i < line_len:
            current = line[i]
            if current == '(':
                # This is the beginning of a node
                if init:
                    # The initial node is done now
                    init = False
                    continue
                if num_children == 0:
                    # Add the left child first
                    temp = Node()
                    i = temp.read(line, i+1, False)
                    self.add_child(temp)
                elif num_children == 1:
                    # Add the right child
                    temp = Node()
                    i = temp.read(line, i+1, False)
                    self.add_child(temp)
                else:
                    # Terminate program, something might be wrong
                    assert False

                # In any case, you did add a child, so...
                num_children += 1

            elif current == ')':
                # This is the end of a node, word is complete
                if word != '':
                    # To avoid the case where you read spaces between nodes
                    from Model import Model
                    # Get vector for this word
                    self.word = word
                    self.vec = Model.get_vec(word)

                assert num_children == 2 or num_children == 0
                # Otherwise, something is wrong
                # Return index of completed word
                return i

            elif current == ' ' or current == '\n' or current == '\t':
                # You have only read the label so far, not the word
                word = ''

            else:
                # Just keep reading, man
                word += current

            # In any case, you did read a character so increment your pointer
            i += 1