"""
Used for trying out different things. Don't take them seriously.
"""
__author__ = 'raza'
import csv
import pickle

import numpy as np


with open('treebank_scores.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file)

exit()

with open('treebank_vectors.pickle', 'rb') as pickle_file:
    d = pickle.load(pickle_file)

for vec in d:
    d[vec] = d[vec][:, np.newaxis]

with open('treebank_vectors.pickle', 'wb') as pickle_file:
    pickle.dump(d, pickle_file, pickle.HIGHEST_PROTOCOL)



data = list(csv.reader(open('Stanford_politeness_corpus/wikipedia.annotated.csv'), delimiter=','))
del data[0]
data = np.array(data)
wiki_lines = data[:, 2]

with open('glove-50d.txt', 'rb') as f:
    glove_lines = f.readlines()

glove_list = []

for line in glove_lines:
    vec = line.split()
    glove_list.append(vec[0])



