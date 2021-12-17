"""
author: Katsuya Lex Colon
updated: 12/03/21
group: Cai Lab
"""
import pandas as pd
import numpy as np

def colocalizing_dots(df, channels=(0,2), hyb = 0, pixel_cutoff=2):
    """Generates a distance matrix by euclidean metric then sorts to get nearest neighbor.
    If the nearest neighbor has a euclidean pixel distance <= pixel_cutoff then the dots are colocalizing.
    Parameters
    ----------
    df = dataframe (hyb,ch,x,y,z,s,w)
    channels = two channels that should colocalize
    hyb = which hyb we are comparing
    pixel_cutoff = upper bound of number of pixels away
    """
    
    #get specific hyb and isolate channel pairs
    hyb_df = df[df["hyb"] == hyb]
    hyb_split_ch1 = hyb_df[hyb_df["ch"]==channels[0]].reset_index(drop=True)
    hyb_split_ch2 = hyb_df[hyb_df["ch"]==channels[1]].reset_index(drop=True)
    
    #convert to numpy array
    arr1 = hyb_split_ch1[["x","y"]].values
    arr2 = hyb_split_ch2[["x","y"]].values

    #generated euclidean distance matrix
    dis_mtx=np.zeros(shape=(len(arr1),len(arr2)))
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            l2 = np.linalg.norm(arr1[i,0:2]-arr2[j,0:2]) 
            dis_mtx[i,j] = l2
    #loop through and return nearest neighbor dot
    #store pairs if they are <= pixel cutoff
    store_pairs = []
    for i in range(len(dis_mtx)):
        nearest_dot_distance = sorted(dis_mtx[i])[0]
        sorted_index = np.argsort(dis_mtx[i])[0]
        if nearest_dot_distance <= pixel_cutoff:
            store_pairs.append([i,sorted_index])
        else:
            continue
            
    #convert pairs to array        
    pairs = np.array(store_pairs)
    
    #colocalization efficiency
    eff = len(store_pairs)/((len(arr1)))
    
    #keep rows with pairs
    hyb_split_ch1 = hyb_split_ch1.iloc[pairs[:,0]]
    hyb_split_ch2 = hyb_split_ch2.iloc[pairs[:,1]]
    
    #combine df
    new_df = pd.concat([hyb_split_ch1,hyb_split_ch2])
    new_df = new_df.reset_index(drop=True)
    
    print("colocalization efficiency =",np.round(eff, 1))
    
    return new_df, np.round(eff, 1)