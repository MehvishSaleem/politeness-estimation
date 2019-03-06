from Node import Node
from Tree import Tree
from utilities import*
import pickle


class Model:
    word_to_vec = None
    targets = None

    def __init__(self, dim=1, reg_cost=0.001, l_rate=0.05, mini_batch=10, epochs=100):
        # list of trees in training set
        self.trees = []

        # training set
        self.tree_train = None

        # validation set
        self.tree_val = None

        # test set
        self.tree_test = None

        # weight matrix for internal nodes
        self.w = init_random(mean=0, var=0.1, shape=(dim, 2*dim))

        # weight matrix for softmax prediction
        self.ws = init_random(mean=0, var=0.1, shape=(2, dim))

        # weight decay
        self.reg_cost = reg_cost

        # learning rate
        self.l_rate = l_rate

        # number of epochs to run
        self.epochs = epochs

        # mini batch size
        self.mini_batch = mini_batch

        # word vector dimension
        self.dim = dim

        # type of activation function
        self.activ_func = "tanh"

        # word-embdeddings dictionary
        with open('treebank_vectors.pickle', 'rb') as pickle_file:
            Model.word_to_vec = pickle.load(pickle_file)

        # target value for each tree
        with open('treebank_scores.pickle', 'rb') as pickle_file:
            Model.targets = pickle.load(pickle_file)

    def add_tree(self, penn_tree, _id):
        """
        Makes a root node, fills up its structure from PTB format and adds it to the list of trees
        """
        tree = Tree(_id=_id)
        _class = Model.targets[_id]
        tree.set_target(_class)
        tree.root = Node()
        tree.root.read(penn_tree, 0, True)
        # Chaipi for life
        tree.root = tree.root.children[0]
        self.trees.append(tree)

    def forward(self, node):
        """
        Runs a forward pass on the whole tree and calculates vectors at each node
        """
        if node.num_child == 0:
            return node.vec

        elif node.num_child == 1:
            return np.tanh(node.children[0].vec)

        elif node.num_child == 2:
            left = self.forward(node.children[0])
            right = self.forward(node.children[1])
            children = concat(left, right)

            node.vec = np.tanh(np.dot(self.w, children))

            return node.vec

    def calc_outputs(self, tree):
        """
        Calls forward prop and calculates predictions from the tree root
        """
        output_vec = self.forward(tree.root)
        tree.predictions = softmax(np.dot(self.ws, output_vec))

    def back_prop(self, node, delta_com, delta_w, delta_ws):
        """
        Back propagates errors from the root node to all the nodes.
        Computers derivatives for all the model parameters
        """
        if node.num_child == 0:
            # TODO: take word vector derivatives
            return
        elif node.num_child == 1:
            return
        elif node.num_child == 2:
            left_vector = node.children[0].vec
            right_vector = node.children[1].vec
            # [x3, p1]
            # TODO: concatenate with bias here
            children = concat(left_vector, right_vector)

            # delta_w = delta_com * [x3, p1]
            delta_w += np.dot(delta_com, children.T)

            # W.T * delta_com * f'([x3, p1])
            delta_down = np.multiply(np.dot(self.w.T, delta_com), tanh_derivative(children))

            left_delta_down = delta_down[:self.dim]
            right_delta_down = delta_down[self.dim: 2 * self.dim]

            self.back_prop(node.children[0], left_delta_down, delta_w, delta_ws)
            self.back_prop(node.children[1], right_delta_down, delta_w, delta_ws)

    def calc_errors(self, tree, delta_w, delta_ws):
        """
        Calls back prop and computes prediction error from the root node.
        """
        # y - t
        diff_class = tree.predictions - tree.target
        # delta_ws = (y - t) * p2
        delta_ws += np.dot(diff_class, tree.root.vec.T)

        # Ws.T * (y - t)
        delta = np.dot(self.ws.T, diff_class)
        # Ws.T * (y - t) * f'(p2)
        delta_node = np.multiply(delta, tanh_derivative(tree.root.vec))

        tree.error = self.get_cost(tree)

        self.back_prop(tree.root, delta_node, delta_w, delta_ws)

    def update(self, delta_w, delta_ws):
        """
        Updates model parameters from the computer derivatives
        """
        self.w -= (self.l_rate * delta_w)
        self.ws -= (self.l_rate * delta_ws)

    def train(self):
        """
        Runs forward and backward passes on the training set.
        Computes errors and errors derivatives and regularizes.
        Updates model parameters.
        Runs till a stopping criterion is not met.
        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(self.w.shape)
        delta_ws = np.zeros(self.ws.shape)
        train_cost = 0

        # early stopping parameters
        min_cost = np.inf
        max_count = 20
        count_down = max_count
        error_factor = 0.001
        train_size = len(self.tree_train)

        for i in xrange(self.epochs):
            # Shuffle training set and create mini batches
            np.random.shuffle(self.tree_train)
            mini_batches = [self.tree_train[i:min(i + self.mini_batch, train_size)]
                            for i in xrange(0, train_size, self.mini_batch)]
            # run SGD for each mini batch
            for mini_batch in mini_batches:
                train_cost = 0
                for t in mini_batch:
                    tree = self.trees[t]
                    # perform calculations
                    self.calc_outputs(tree)
                    self.calc_errors(tree, delta_w, delta_ws)
                    train_cost += self.get_cost(tree)

                # scale and regularize the parameters
                scale = 1. / len(mini_batch)
                self.scale_regularize(delta_w, delta_ws, scale)
                self.update(delta_w, delta_ws)

                # check performance on validation set for early stopping
                pred_cost = self.validate()
                if pred_cost < (1 - error_factor) * min_cost:
                    min_cost = pred_cost
                    count_down = max_count
                else:
                    count_down -= 1

                # performance on validation set has not decreased significantly in the past
                if count_down == 0:
                    break

                # Reset the derivatives
                delta_w.fill(0)
                delta_ws.fill(0)
        return train_cost

    def validate(self):
        """
        Computes and returns prediction accuracy on the validation set
        """
        val_cost = 0
        for t in self.tree_val:
            tree = self.trees[t]
            self.calc_outputs(tree)
            val_cost += self.get_cost(tree)

        return val_cost

    def test(self):
        """
        Computes and returns predictions on the hand-held test set.
        Also Returns the number of correct predictions made and the
        ids of incorrectly predicted trees
        """
        test_cost = 0
        correct = 0
        incorrect = []
        for t in self.tree_test:
            tree = self.trees[t]
            self.calc_outputs(tree)
            test_cost += self.get_cost(tree)
            tree.pred_label = np.argmax(tree.predictions)
            true_label = np.where(tree.target == 1)[0]
            if true_label == tree.pred_label:
                correct += 1
            else:
                incorrect.append(tree.id)

        return np.around(test_cost, 3), 1.*correct/len(self.tree_test), incorrect

    def check_model_veracity(self):
        """
        Checks whether the model is correct by performing numerical gradient check.
        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(self.w.shape)
        delta_ws = np.zeros(self.ws.shape)
        for i in xrange(self.epochs):
            numgrad = None
            for t in self.tree_train:
                tree = self.trees[t]
                self.calc_outputs(tree)
                self.calc_errors(tree, delta_w, delta_ws)
                if numgrad is not None:
                    numgrad += self.numerical_gradient(tree)
                else:
                    numgrad = self.numerical_gradient(tree)

            scale = 1. / len(self.tree_train)
            numgrad *= scale
            self.scale_regularize(delta_w, delta_ws, scale)
            grad = self.get_gradients(delta_w, delta_ws)
            print np.around(np.sum(np.abs(grad - numgrad) / np.abs(grad + numgrad)), 10)

            self.update(delta_w, delta_ws)
            delta_w.fill(0)
            delta_ws.fill(0)

    def scale_regularize(self, delta_w, delta_ws, scale):
        """
        Performs regularization of the cost function.
        L2 regularization with weight decay
        """
        delta_w *= scale
        delta_w += (self.reg_cost * self.w)
        delta_ws *= scale
        delta_ws += (self.reg_cost * self.ws)

    def get_cost(self, tree):
        """
        Computes the Cross Entropy cost function with regularization.
        Uses computed predictions from the tree.
        """
        # Summation {t * log(y)}
        _log = np.log(tree.predictions)
        # TODO: remove the above line and uncomment the line below
        # _log = np.where(tree.predictions > 0, np.log(tree.predictions), 0)
        cost = -(np.sum(np.multiply(tree.target, _log)))
        # L2 weight decay
        cost += self.reg_cost / 2 * (np.sum(self.w * self.w))
        cost += self.reg_cost / 2 * (np.sum(self.ws * self.ws))

        return cost

    def get_params(self):
        """
        Concatenates all model parameters into one-dimensional vector and returns.
        """
        w_ = np.reshape(np.ravel(self.w), (-1, 1))
        ws_ = np.reshape(np.ravel(self.ws), (-1, 1))
        params = np.vstack((w_, ws_))

        return params

    def set_params(self, new_params):
        """
        Sets all the model parameters in a one-dimensional vector
        """
        self.w = np.reshape(new_params[:self.dim * 2*self.dim], self.w.shape)
        self.ws = np.reshape(new_params[self.dim * 2*self.dim:], self.ws.shape)

    def numerical_gradient(self, tree):
        """
        Performs numerical gradient checking by taking derivative by definition.
        See Stanford UFLDL for theoretical details.
        """
        epsilon = 1e-5
        initial_params = self.get_params()
        self.set_params(initial_params)
        l = len(initial_params)
        vector = np.zeros(initial_params.shape)
        exp_grad = np.zeros(initial_params.shape)

        for i in range(l):
            vector[i] = epsilon

            self.set_params(initial_params + vector)
            self.calc_outputs(tree)
            c_plus = self.get_cost(tree)

            self.set_params(initial_params - vector)
            self.calc_outputs(tree)
            c_minus = self.get_cost(tree)

            exp_grad[i] = (c_plus - c_minus) / (2 * epsilon)

            vector[i] = 0

        self.set_params(initial_params)

        return exp_grad

    @staticmethod
    def get_gradients(delta_w, delta_ws):
        """
        Concatenates the derivatives of all model parameters and returns.
        """
        deltaw_ = np.reshape(np.ravel(delta_w), (-1, 1))
        deltaws_ = np.reshape(np.ravel(delta_ws), (-1, 1))

        return np.vstack((deltaw_, deltaws_))

    @staticmethod
    def get_vec(word):
        """
        Maps word to its vector from the embedding matrix
        """
        if word in Model.word_to_vec:
            return Model.word_to_vec[word]
        else:
            return Model.word_to_vec['unknown']
