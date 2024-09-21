import networkx as nx
import pandas as pd
import numpy as np 
import heapq
import json
import pickle
from tqdm import tqdm
from collections import namedtuple

csv_file1 = "type12_type34/type12.csv"
csv_file2 = "type12_type34/type34.csv"

topo = []
vis = {}
edges = {}

def dfs(u):
    global topo
    global vis
    global edges
    
    vis[u] = True
    
    for v in edges[u]:
        if not vis[v]:
            dfs(v)
    topo.append(u)

def build_graph(file_paths):
    G = nx.MultiDiGraph()
    added_nodes = set()

    for file_path in file_paths:
        with open(file_path, "r") as file:      
            for line in tqdm(file, desc=f"Processing {file_path}"):
                data = line.strip().split(",")[:-2]
                
                source_id, target_id = int(data[0]), int(data[4])
                
                if source_id not in added_nodes:
                    G.add_node(source_id, 
                               route_id=int(data[1]),
                               var_id=int(data[2]),
                               timestamp=float(data[3]),
                               node_type=int(data[15]),
                               latx=float(data[9]),
                               lngy=float(data[10]))
                    added_nodes.add(source_id)
                
                if target_id not in added_nodes:
                    G.add_node(target_id, 
                               route_id=int(data[5]),
                               var_id=int(data[6]),
                               timestamp=float(data[7]),
                               node_type=int(data[16]),
                               latx=float(data[11]),
                               lngy=float(data[12]))
                    added_nodes.add(target_id)
                
                # Add edge
                number_transfer = 0 if data[1] == data[5] else 1
                G.add_edge(source_id, target_id,
                           weight=(number_transfer, float(data[8])),
                           route_id_x=int(data[1]),
                           var_id_x=int(data[2]),
                           route_id_y=int(data[5]),
                           var_id_y=int(data[6]),
                           edge_departure=float(data[3]),
                           edge_arrival=float(data[7]),
                           edge_type=int(data[20]),
                           edge_pos=int(data[19]))

    return G

def dijkstra_raw(G, start):
    distances = {node: (float('infinity'), float('infinity'), float('infinity'), float('infinity')) for node in G}
    distances[start] = (0, 0, G.nodes[start]['timestamp'], G.nodes[start]['timestamp'])
    pq = [(0, 0, G.nodes[start]['timestamp'], G.nodes[start]['timestamp'], start, None)]
    paths = {node: [] for node in G}
    paths[start] = [start]
    edge_keys = {node: None for node in G}

    while pq:
        transfers, time_diff, dep_time, arr_time, node, prev_edge = heapq.heappop(pq)

        if (transfers, time_diff, dep_time, arr_time) > distances[node]:
            continue

        for neighbor, edges in G[node].items():
            for edge_key, edge_data in edges.items():
                new_transfers = transfers + edge_data['weight'][0]
                new_time_diff = time_diff + edge_data['weight'][1]
                new_arr_time = G.nodes[neighbor]['timestamp']

                new_cost = (new_transfers, new_time_diff, dep_time, new_arr_time)

                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    paths[neighbor] = paths[node] + [neighbor]
                    edge_keys[neighbor] = edge_key
                    heapq.heappush(pq, (*new_cost, neighbor, edge_key))

    return distances, paths, edge_keys

# OPTIMIZATION STAGE

def get_cost(edge_data):
    """Calculate the cost of an edge based on the given criteria."""
    
    """ 
    number_of_transfers < time_difference
    earliest_departure_time < earlies_arrival_time
    """ 
    
    number_of_transfers, time_difference = edge_data['weight']
    edge_departure = edge_data['edge_departure']
    edge_arrival = edge_data['edge_arrival']
    
    return (number_of_transfers, time_difference, edge_departure, edge_arrival)

def optimize_graph(G):
    optimized_G = nx.DiGraph()
    
    for node, node_data in G.nodes(data=True):
        optimized_G.add_node(node, **node_data)

    for node in tqdm(G.nodes(), desc="Optimizing graph"):
        neighbors = list(G[node])
        for neighbor in neighbors:
            edges = G.get_edge_data(node, neighbor)
            best_edge_key, best_edge_data = min(edges.items(), key=lambda item: get_cost(item[1]))
            optimized_G.add_edge(node, neighbor, **best_edge_data)
            
            # Change attributes of nodes according on optimized edge
            
            G.nodes[node]['timestamp'] = best_edge_data['edge_departure']
            G.nodes[node]['route_id'] = best_edge_data['route_id_x']
            G.nodes[node]['var_id'] = best_edge_data['var_id_x']
            
            G.nodes[neighbor]['timestamp'] = best_edge_data['edge_arrival']
            G.nodes[neighbor]['route_id'] = best_edge_data['route_id_y']
            G.nodes[neighbor]['var_id'] = best_edge_data['var_id_y']

    return optimized_G

def dijkstra_one(G, source):
    distances = {node: (float('infinity'), float('infinity'), float('infinity'), float('infinity')) for node in G.nodes()}
    distances[source] = (0, 0, G.nodes[source]['timestamp'], G.nodes[source]['timestamp'])
    pq = [(0, 0, G.nodes[source]['timestamp'], G.nodes[source]['timestamp'], source)]
    paths = {node: [] for node in G}
    paths[source] = [source]
    visited = set()
    
    while pq:
        transfers, time_diff, dep_time, arr_time, node = heapq.heappop(pq)
        
        if (transfers, time_diff, dep_time, arr_time) > distances[node]:
            continue
        
        visited.add(node)
        
        for neighbor, edges in G[node].items():
            if neighbor == node or neighbor in visited:
                continue
            
            if edges['edge_arrival'] < dep_time: 
                continue
            
            # print(neighbor)
            
            new_transfers = transfers + (1 if edges['route_id_x'] != edges['route_id_y'] else 0)
            new_time_diff = time_diff + edges['weight'][1]
            new_arr_time = edges['edge_arrival']
            
            new_cost = (new_transfers, new_time_diff, dep_time, new_arr_time)
            
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                paths[neighbor] = list(paths[node]) + [neighbor]
                heapq.heappush(pq, (*new_cost, neighbor))
                
    return distances, paths

def dijkstra_all(G):
    all_dist = {}
    all_paths = {}
    
    for source in tqdm(G.nodes(), desc="Computing Dijkstra for all nodes"):
        distances, paths = dijkstra_one(G, source=source)
        all_dist[source] = distances
        all_paths[source] = paths
        
    return all_dist, all_paths

def save_all_shortest_paths(all_dist, filename="output/all_shortest_paths.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(all_dist, file)
        
def load_all_shortest_paths(filename="output/all_shortest_paths.pkl"):
    with open(filename, "rb") as file:
        all_dist = pickle.load(file)
    
    return all_dist

def dijkstra_all_for_counting_importance(G):
    all_dist = {}
    all_paths = {}
    cnt = {}
    
    for i in G.nodes():
        cnt[i] = {}
        for j in G.nodes():
            cnt[i][j] = 0
        
    for source in tqdm(G.nodes(), desc="Computing Dijkstra for K importance"):
        distances = {node: (float('infinity'), float('infinity'), float('infinity'), float('infinity')) for node in G.nodes()}
        distances[source] = (0, 0, G.nodes[source]['timestamp'], G.nodes[source]['timestamp'])
        pq = [(0, 0, G.nodes[source]['timestamp'], G.nodes[source]['timestamp'], source)]
        paths = {node: [] for node in G}
        paths[source] = [source]
        visited = set()
        
        cnt[source][source] = 1
        
        while pq:
            transfers, time_diff, dep_time, arr_time, node = heapq.heappop(pq)
            
            if (transfers, time_diff, dep_time, arr_time) > distances[node]:
                continue
            
            visited.add(node)
            
            for neighbor, edges in G[node].items():
                if neighbor == node or neighbor in visited:
                    continue
                
                if edges['edge_arrival'] < dep_time: 
                    continue
                
                # print(neighbor)
                
                new_transfers = transfers + (1 if edges['route_id_x'] != edges['route_id_y'] else 0)
                new_time_diff = time_diff + edges['weight'][1]
                new_arr_time = edges['edge_arrival']
                
                new_cost = (new_transfers, new_time_diff, dep_time, new_arr_time)
                
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    paths[neighbor] = list(paths[node]) + [neighbor]
                    heapq.heappush(pq, (*new_cost, neighbor))
                    cnt[source][neighbor] = cnt[source][node] # count importance
                elif (new_transfers, new_time_diff) == (distances[neighbor][0], distances[neighbor][1]):
                    cnt[source][neighbor] += cnt[source][node] # count importance
        
        all_dist[source] = distances
        all_paths[source] = paths

    return all_dist, all_paths, cnt

def top_k_importance(G, cnt, all_dist):
    global topo
    global vis
    global edges
    
    impo = {}
    
    for i in G.nodes():
        impo[i] = 0
    
    for st in tqdm(G.nodes(), desc="Counting top K important stops"):
        topo = []
        
        for i in G.nodes():
            edges[i] = []
            vis[i] = False
        
        # Remove all edges which are not shortest paths
        for i in G.nodes():
            for j in G.neighbors(i):
                if i in all_dist[st] and j in all_dist[st]:
                    if (all_dist[st][i][1] + G[i][j]['weight'][1] == all_dist[st][j][1]):
                        edges[i].append(j)
                else:
                    # Print missing keys for debugging
                    if i not in all_dist[st]:
                        print(f"Node {i} is missing in all_dist[{st}]")
                    if j not in all_dist[st]:
                        print(f"Node {j} is missing in all_dist[{st}]")
                    break
                    
        for i in G.nodes():
            if not vis[i]:
                dfs(i)
                
        dp = {}
        
        for i in G.nodes():
            dp[i] = 0
        
        for i in topo:
            dp[i] = 1
            for j in edges[i]:
                dp[i] += dp[j]
                
            cnt[st][i] *= dp[i]
        
    for v in G.nodes():
        for u in G.nodes():
            impo[v] += cnt[u][v]
    
    return impo


    
    

        

    




    

































