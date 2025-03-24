"""
@kcolon
date: 250117
"""

#general analysis packages
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

def colocalizing_dots_within(df, ref_hyb=1, radius=1, neighbors_expected=5):
    # Reset index for df
    df = df.reset_index(drop=True)
    
    # Initialize NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=neighbors_expected, radius=radius, metric="euclidean", n_jobs=-1)
    
    # Selecting the initial seed points from the reference hybridization round
    initial_seed = df[df.hyb == ref_hyb]

    # DataFrame excluding the reference hybridization round
    removed_reference = df[df.hyb != ref_hyb]
    
    # Lists to store neighbors and distances
    neighbor_list = []
    distance_list = []

    for hyb in removed_reference.hyb.unique():
        # Fit NearestNeighbors on the current hyb round
        current_hyb_points = df[df.hyb == hyb][["x", "y"]]
        neigh.fit(current_hyb_points)

        # Find neighbors for the initial seed points
        distances, neighbors = neigh.radius_neighbors(initial_seed[["x", "y"]], radius=radius, 
                                                      return_distance=True, sort_results=True)
        distance_list.append(distances)
        neighbor_list.append(neighbors)

    # Flattening the lists and grouping reference dots with colocalizing pairs
    neighbor_list2 = []
    distance_list2 = []
    for i in range(len(neighbor_list[0])):
        temp = []
        temp_dist = []
        for j in range(len(neighbor_list)):
            if len(neighbor_list[j][i]) == 0:
                continue
            else:
                temp.extend([[neighbor_list[j][i][0]]])
                temp_dist.extend([[distance_list[j][i][0]]])

        orig_idx = [initial_seed.iloc[i].name]
        for hyb, idx_list in enumerate(temp):
            for idx in idx_list:
                try:
                    orig_idx.append(df[df.hyb == removed_reference.hyb.unique()[hyb]].iloc[idx].name)
                except IndexError:
                    continue

        neighbor_list2.append(orig_idx)
        distance_list2.append(temp_dist)
    
    #filter dots that are not colocalizing completely across all hybs
    filtered_neigh = []
    filtered_dist  = []
    for i in range(len(neighbor_list2)):
        if len(neighbor_list2[i]) < neighbors_expected:
            continue
        else:
            filtered_neigh.append(neighbor_list2[i])
            filtered_dist.append([ele for sublist in distance_list2[i] for ele in sublist])

    return filtered_neigh, filtered_dist

def colocalizing_dots_across_channels(df, ref_hyb=1, channel1=1, channel2=2, 
                                      radius=1, neighbors_expected=5, cutoff = 0.05):
    # Reset index for df
    df = df[df["max intensity"] > cutoff]
    df = df.reset_index(drop=True)
    
    # Initialize NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=neighbors_expected, radius=radius, metric="euclidean", n_jobs=-1)
    
    # Selecting the seed points from the reference hybridization round for both channels
    channel1_seed = df[(df.hyb == ref_hyb) & (df.ch == channel1)]
    channel2_seed = df[(df.hyb == ref_hyb) & (df.ch == channel2)]

    # List to store neighbors and distances
    neighbor_list = []
    distance_list = []

    # Fit NearestNeighbors on channel2 points
    neigh.fit(channel2_seed[["x", "y"]])

    # Find neighbors for the channel1 seed points
    distances, neighbors = neigh.radius_neighbors(channel1_seed[["x", "y"]], radius=radius, 
                                                  return_distance=True, sort_results=True)
    distance_list.append(distances)
    neighbor_list.append(neighbors)

    # Flattening the lists and grouping channel1 points with their colocalizing pairs in channel2
    colocalized_pairs = []
    for i in range(len(neighbor_list[0])):
        if len(neighbor_list[0][i]) == 0:
            continue
        else:
            channel1_index = channel1_seed.iloc[i].name
            channel2_index = channel2_seed.iloc[neighbor_list[0][i][0]].name
            colocalized_pairs.append((channel1_index, channel2_index, distance_list[0][i][0]))
                
    df_final = pd.DataFrame(colocalized_pairs)
    df_final.columns = ["index 1", "index 2", "Distance"]
    
    return df_final, np.round(len(df_final)/len(channel1_seed),2)