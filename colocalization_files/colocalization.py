"""
author: Katsuya Lex Colon
updated: 02/14/21
group: Cai Lab
"""
#general analysis packages
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def colocalizing_dots(df1, df2, radius=1, return_dots_not_coloc=False):
    """
    Performs nearest neighbor search provided a given search radius.
    If the nearest neighbor has a euclidean pixel distance <= radius then the dots are colocalizing.
    Parameters
    ----------
    df1 = first set of dots
    df2 = second set of dots
    radius = search radius
    return_dots_not_coloc = bool to return dots not colocalizing
    """
    
    #reset index for df just in case
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    #using sklearn nearest neighbor algorithm to find nearest dots
    #initialize algorithm
    neigh = NearestNeighbors(n_neighbors=2, radius=radius, metric="euclidean", n_jobs=1)
    
    #initialize neighbor
    initial_seed = df1[["x","y"]]
    #find neighbors for df1
    neigh.fit(df2[["x","y"]])
    distances,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
    
    #nearest neighbor dot
    neighbors_flattened = []
    for i in range(len(neighbors)):
        try:
            neighbors_flattened.append([i,neighbors[i][0]])
        except IndexError:
            continue
            
    #keep dots that colocalize
    new_df1 = df1.iloc[np.array(neighbors_flattened)[:,0]].reset_index(drop=True)
    new_df2 = df2.iloc[np.array(neighbors_flattened)[:,1]].reset_index(drop=True)
    
    #colocalization efficiency
    eff = len(new_df1)/len(df1)
    
    print("colocalization efficiency =",np.round(eff, 2))
   
    if return_dots_not_coloc == True:
        #separate file for dots that do not colocalize
        df1_nocoloc_idx = np.array(list(set(df1.index)-set(np.array(neighbors_flattened)[:,0]))) 
        df1_nocoloc = df1.iloc[df1_nocoloc_idx].reset_index(drop=True)
        df2_nocoloc_idx = np.array(list(set(df2.index)-set(np.array(neighbors_flattened)[:,1])))
        df2_nocoloc = df2.iloc[df2_nocoloc_idx].reset_index(drop=True)
        
        return eff, [df1_nocoloc, df2_nocoloc]
    else:
        #return new dfs that do colocalize
        return eff, [new_df1, new_df2]
    
    
    
    