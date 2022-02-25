from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def neighbor_search(locations, hybs=12, num_barcodes=4, seed=0, radius=1):
    """
    Find neighbors across barcoding rounds
    
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

    #store index of barcoding round that is being compared against in list
    initial_barcoding_index = barcoding_round[seed][["x","y"]].index.tolist()

    #rename seed
    initial_seed = barcoding_round[seed][["x","y"]]

    #delete barcoding round that is being compared against in list
    del barcoding_round[seed]

    #find neighbors at a given search radius between barcoding rounds
    neighbor_list = []

    for i in range(len(barcoding_round)):
        #initialize neighbor
        neigh.fit(barcoding_round[i][["x","y"]])
        #find neighbors for barcoding round 0
        _,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
        neighbor_list.append(neighbors)
    
    return neighbor_list

def count_neighbors(neighbor_list):
    """
    counts neighbors across barcoding rounds
    
    Parameters
    ---------
    neighbor_list = list of neighbors outputted from neighbor_search()
    
    """    
    
    neigh_count_list = []
    
    for brcd_round in neighbor_list:
        #get number of neighbors per dot
        neighbor_count = [len(brcd_round[i]) for i in range(len(brcd_round))]
        #get average neighbor count
        avg_neighbor = np.mean(neighbor_count)
        #add average to list
        neigh_count_list.append(avg_neighbor)
    
    #return average neighbor for each round
    return np.mean(neigh_count_list)