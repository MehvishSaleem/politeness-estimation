from Model import Model
import numpy as np
import pickle


if __name__ == '__main__':
    #####################################
    # don't change this
    np.random.seed(43)
    # hyper parameters to be tweaked here
    load_rnn_from_pickle = False
    training_size = 10
    l_rate = 0.05
    mini_batch_size = 20
    reg_cost = 0.001
    epochs = 100
    ######################################

    # perform 60-15-25 percent split into train-val-test set
    train = np.ceil(0.6 * training_size)
    val = np.ceil(0.15 * training_size)
    test = training_size - train - val
    dim = 50

    # load parsed trees from file in PTB format
    if load_rnn_from_pickle is False:
        with open('WikiTreebankQuartiles_second.txt', 'rb') as fh:
            RNN = Model(dim=dim, l_rate=l_rate, mini_batch=mini_batch_size, reg_cost=reg_cost, epochs=epochs)

            for i, line in enumerate(fh):
                p = line.find(' ')
                ptb_string = line[p + 1:]
                rid = line[:p]
                # Add to the list of trees
                RNN.add_tree(ptb_string, rid)
        """
        with open('rnn.pickle', 'wb') as pickle_file:
            pickle.dump(RNN, pickle_file, pickle.HIGHEST_PROTOCOL)
        """
    else:
        with open('rnn.pickle', 'rb') as pickle_file:
            RNN = pickle.load(pickle_file)

    indices = np.arange(0, training_size)
    # create separate indices for the 3 data sets
    np.random.shuffle(RNN.trees)
    np.random.shuffle(indices)
    RNN.tree_train = indices[:train]
    RNN.tree_val = indices[train:train + val]
    RNN.tree_test = indices[train + val:]
    RNN.train()
    # RNN.check_model_veracity()
    print "Test Cost Function, Accuracy, Incorrectly classified sentence Ids"
    print RNN.test()
    hyper_params = "training_size={0}\nl_rate={1}\nmini_batch_size={2}\nreg_cost={3}\nepochs={4}".format(
        training_size, l_rate, mini_batch_size, reg_cost, epochs)
    print hyper_params