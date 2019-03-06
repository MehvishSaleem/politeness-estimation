import numpy as np


class Tree:
    def __init__(self, _id):
        # id as given in the corpus
        self.id = _id
        self.root = None
        self.target = np.zeros(shape=(2, 1))
        self.predictions = None
        self.pred_label = None
        self.error = 0

    def set_target(self, _class):
        self.target[_class] = 1