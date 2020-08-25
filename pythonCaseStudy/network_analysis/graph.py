import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3, 4, 5, 6, 7, 8,9 , 'sipe', 'spencer', 'sipsip'])
G.add_edge(1, 2)
G.add_edges_from([(1, 2), (3, 4), (5,6), (7,8), (1, 9), ('sipe', 'spencer'), (9, 8)])

karate = nx.karate_club_graph()
pos = nx.spring_layout(karate)
plt.clf()
nx.draw_networkx(karate, pos = pos, with_labels = True, node_color = 'lightblue', edge_color = 'gray')
nx.draw_networkx_labels(karate, pos = pos, with_labels = True, node_color = 'lightblue', edge_color = 'gray')
nx.draw_networkx_edges(karate, pos = pos, with_labels = True)
plt.show()