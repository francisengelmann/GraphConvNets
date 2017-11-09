import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import utils

G = nx.house_graph()
pos={0:(0,0),
     1:(1,0),
     2:(0,1),
     3:(1,1),
     4:(0.5,2.0)}
adj = nx.adjacency_matrix(G)
adj = adj.todense()

print('Adjacency')
print(adj)
N = adj.shape[0]
e = np.zeros((N,1))
e[0, 0] = 1
e[1, 0] = 10
e[2, 0] = 100
e[3, 0] = 1000
e[4, 0] = 10000

# Compute degree matrix using adj matrix
# Adj*e = diagnoal entriies of each node
o = np.ones((N,1))
print("Degree")
print(np.matmul(adj, o))

print(np.matmul(adj, e))

print(np.matmul(adj, np.identity(N)))


plt.figure(figsize=(4, 4))
nx.draw(G,
        cmap=plt.get_cmap('jet'),
        # node_color='',
        labels={i: i for i in range(G.number_of_nodes())},
        font_color='white',
        pos=pos)
plt.show()
exit()


degree = np.zeros(np.shape(adj))
for v in G:
    degree[v, v] = G.degree(v)

plt.axis('on')

pred_labels_dict = {i: i for i in range(adj.shape[0])}

nx.draw(G, cmap=plt.get_cmap('jet'), node_color=labels_gt, labels=pred_labels_dict, font_color='white')
plt.axis('on')
plt.show()