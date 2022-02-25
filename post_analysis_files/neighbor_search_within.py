"""
author: Katsuya Lex Colon
updated: 02/14/21
group: Cai Lab
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def neighbor_search_within(locations, hybs=12, num_barcodes=4, seed=0, radius=1):
    
    """
    
    Find neighbors within barcoding rounds
    
    Parameters
    ---------
    locations = locations.csv file
    hybs = total number of hybs
    num_barcodes = total number of barcodes
    seed = which round to look at first
    radius = distance by pixels
    
    """
    
    #initialize neighbor
    neigh = NearestNeighbors(n_neighbors=num_barcodes, radius=radius, metric="euclidean", n_jobs=1)
    barcoding_round = []
    #separate locations by barcoding round
    hyb_rounds = np.arange(0,hybs,1)
    temp = []
    for h in hyb_rounds:
        if h == hyb_rounds[len(hyb_rounds)-1]:
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)
            comp_round = pd.concat(temp)
            barcoding_round.append(comp_round) 
        elif (h % (hybs/num_barcodes) != 0) or (h == 0):
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)
        else:
            comp_round = pd.concat(temp)
            barcoding_round.append(comp_round)
            temp = []
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)

    #remove temp list
    del temp

    #rename seed
    initial_seed = barcoding_round[seed][["x","y"]]

    #delete rest of the barcoding rounds
    del barcoding_round
    
    #initialize neighbor
    neigh.fit(initial_seed)
    
    #find neighbors for within same barcoding round
    _,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
    
    return neighbors, initial_seed

def density_estimate(neighbors, radius=1, average=False):
    """
    estimate density of dots provided a search radius
    
    Parameters
    ----------
    neighbors = list of neighbors from neighbor_search_within()
    radius = search radius in pixels
    average = calculate the average density or return individual values
    
    """
    
    neighbors_count =[]
    index = []
    for i in range(len(neighbors)):
        #ignore first neighbor since that is self
        if len(neighbors[i]) == 1:
            neighbors_count.append(0)
        else:
            neighbors_count.append(len(neighbors[i])-1)
            index.append([neighbors[i]])
    #get density by dividing by pixel area
    area = np.pi*(radius**2)
    density = np.array(neighbors_count)/area
    
    #convert to df
    density = pd.DataFrame(density)
    density["radius"] = f"Radius = {radius}"
    density.columns = ["number of neighbors/pixel","radius"]
    
    if average == True:
        #get average pixel density 
        area = np.pi*(radius**2)
        mean = np.mean(neighbors_count)
        density = mean/area
        
    #return average neighbor for each round
    return density, index
