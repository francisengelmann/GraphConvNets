import networkx as nx
import numpy as np
import community

def load_data_karate():
    G = nx.karate_club_graph()

    adj = nx.adjacency_matrix(G)
    adj = adj.todense()

    feats = np.zeros(np.shape(adj))
    print("Node Degree")
    for v in G:
        feats[v,v] =  G.degree(v)
        #print('%s %s' % (v, G.degree(v)))

    # 2 classes
    labels_2 = [
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
<<<<<<< Updated upstream
    return adj, feats, labels
=======
    labels_2 = np.squeeze([x - 1 for x in labels_2])

    # 4 classes
    if False: # recompute 4 classes
        labels_4 = community.best_partition(G).items()
        labels_4 = [x[1] for x in labels_4]
    else:
        labels_4 = [0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 2, 0, 2, 0, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 2]
    return adj, feats, labels_4, G

def load_data_3DSMR():
    pass
>>>>>>> Stashed changes
