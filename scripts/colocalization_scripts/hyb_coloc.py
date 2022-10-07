"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 06/07/2022
"""

#custom function
import tifffile as tf
from daostarfinder_dotdetection import *
#general analysis packages
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
#parallel processing
from concurrent.futures import ProcessPoolExecutor
#file management
import os

def colocalizing_dots(df1, df2, radius=1):
    """
    Performs nearest neighbor search provided a given search radius.
    If the nearest neighbor has a euclidean pixel distance <= radius then the dots are colocalizing.
    Parameters
    ----------
    df1 = first set of dots
    df2 = second set of dots
    radius = search radius
    """
    
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
            
    #colocalization efficiency
    eff = round((len(new_df1)/len(df1)),3)
    
    return eff

def last_hyb_coloc(img_src_1, img_src_2, channel=1, z = 0, pos=0, threshold= 0.02, hyb_list = [0,48],
                   radii_list = [0.75,1,2], num_channels=4):
    """
    This function will return colocalization eff between hybs. 
    
    Parameters
    ----------
    img_src_1: path to 1st image
    img_src_2: path to 2nd image
    channel: which channel to detect spots (1,2,3,4,5)
    z: which z pos to look at (0,1,2...)
    pos: which pos
    threshold: threshold value for pixel intensity
    radii_list: list of various radii
    pos: which position to look at
    hyb_list: which two hybs should colocalize
    swapaxes: bool to flip z and channel axes.
    """
    
    #detect spots
    dots_1 = dot_detection(img_src_1, HybCycle=hyb_list[0], size_cutoff=None, 
                           threshold=threshold, channel=channel, num_channels=num_channels)
    dots_2 = dot_detection(img_src_2, HybCycle=hyb_list[1], size_cutoff=None, 
                           threshold=threshold, channel=channel, num_channels=num_channels)
    #isolate z
    dots_1 = dots_1[dots_1["z"]==z].reset_index(drop=True)
    dots_2 = dots_2[dots_2["z"]==z].reset_index(drop=True)

    #calculate colocalization at various radii
    colocal_list = []
    for radii in radii_list:
        eff = colocalizing_dots(dots_1, dots_2, radius=radii)
        colocal_list.append([channel,pos,eff,radii])

    #convert to df then combine
    final_df = pd.DataFrame(colocal_list)
    final_df.columns = ["ch","pos","eff","radii"]
    
    return final_df

def coloc_parallel(img_dir = None, channel=1, z = 0, threshold= 0.02, 
                   radii_list = [0.75,1,2], num_pos=25,
                   hyb_list = [0,48], num_channels=4):
    
    """
    Run colocalization in parallel for each pos.
    Parameters
    ----------
    img_dir: path to image directory
    channel: which channel to detect spots (1,2,3,4,5)
    z: which z pos to look at (0,1,2...)
    threshold: threshold value for pixel intensity
    radii_list: list of various radii
    num_pos: how many pos
    hyb_list: which two hybs should colocalize
    swapaxes: bool to flip z and channel axes.
    """
    #set output paths
    parent = Path(img_dir)
    while "seqFISH_datapipeline" not in os.listdir(parent):
        parent = parent.parent
    output_path = parent / "seqFISH_datapipeline"/ "hyb_colocalization"/ f"hyb_coloc_channel_{channel}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #coloc for multiple pos in parallel
    with ProcessPoolExecutor(max_workers=32) as exe:
        futures = []
        for pos in range(num_pos):
            #image sources
            img_src_1 = str(Path(img_dir) / f"HybCycle_{hyb_list[0]}" / f"MMStack_Pos{pos}.ome.tif")
            img_src_2 = str(Path(img_dir) / f"HybCycle_{hyb_list[1]}" / f"MMStack_Pos{pos}.ome.tif")
            #check if images exist
            try: 
                tf.imread(img_src_1)
                tf.imread(img_src_2)
            except:
                print(img_src_1)
                print(img_src_2)
                print("check if these two paths are correct.")
                continue
            #dot detect
            fut = exe.submit(last_hyb_coloc, img_src_1=img_src_1, img_src_2=img_src_2, 
                             channel=channel, z=z, pos=pos, threshold=threshold, hyb_list=hyb_list,
                             radii_list=radii_list, num_channels=num_channels)
            futures.append(fut)
            
    #collect result from futures objects
    result_list = [fut.result() for fut in futures]
    
    #combine df
    final_df = pd.concat(result_list).reset_index(drop=True)
    final_df.columns = ["ch","pos","eff","radii"]
    final_df.to_csv(str(output_path))
        