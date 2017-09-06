import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np

G = nx.karate_club_graph()
#G = nx.petersen_graph()

#parts = community.best_partition(G)
#values = [parts.get(node) for node in G.nodes()]
#exit(0)

#nx.draw(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')

# Find modularity
#part = community.best_partition(G)
#mod = community.modularity(part,G)

# Plot, color nodes using community structure
# values = [part.get(node) for node in G.nodes()]
# nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)
# plt.show()

nx.drawing.nx_pylab.draw_spring(G)
# nx.drawing.nx_pylab.draw_spectral(G)
plt.show()
print("Node Degree")

for v in G:
    print('%s %s' % (v, G.degree(v)))

adj = nx.adjacency_matrix(G)
print(adj)
print(type(adj))
exit()

N = 100
r0 = 0.6
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)
area = np.pi * (10 * np.random.rand(N))**2  # 0 to 10 point radii
c = np.sqrt(area)
r = np.sqrt(x * x + y * y)
plt.scatter(x, y, marker='o', c=c)

