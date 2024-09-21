from graph import *

G = build_graph([csv_file1, csv_file2])
G = optimize_graph(G)

all_dist, all_paths = dijkstra_all(G)

source = list(G.nodes())[0]  
des = list(G.nodes())[15]

print(f"Shortest path from {source} to {des}: {all_paths[source][des]}")
print(f"Total transfers: {all_dist[source][des][0]}")
print(f"Total time difference: {all_dist[source][des][1]}")


    

