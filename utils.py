import networkx as nx
import numpy as np


def load_data_karate():
    G = nx.karate_club_graph()

    adj = nx.adjacency_matrix(G)
    adj = adj.todense()

    feats = np.zeros(np.shape(adj))
    print("Node Degree")
    for v in G:
        feats[v,v] =  G.degree(v)
        #print('%s %s' % (v, G.degree(v)))


    labels = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        2,
        1,
        2,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    return adj, feats, labels
