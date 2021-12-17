"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 12/03/21
"""

import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def keep_dots_in_cells(mask, dot_locations):
    """a function to remove any dots outside of mask
    Parameter
    ---------
    mask = cellpose generated mask path
    dot_locations = dot_locations path
    
    Returns
    -------
    output locations.csv 
    
    """
    orig_image_dir = Path(dot_locations).parent.parent
    output_folder = Path(orig_image_dir)
    output_path = output_folder / 'genes_in_cells' /Path(dot_locations).relative_to(orig_image_dir).parent / "gene_locations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read in data
    locations = pd.read_csv(dot_locations)
    #cellpose mask outputs (c,y,x)
    img = tf.imread(mask)
    #get x and y coordinates
    locations_xy = locations[["x","y"]].values.astype(int)
    dot_info = []
    #keep dots only in cells
    for i in range(len(locations)):
        x = locations_xy[i][0]
        y = locations_xy[i][1]
        if img[y,x] == 0:
            continue
        else:
            cell = img[y,x]
            dot_info.append([i,cell])
            
    dot_info = np.array(dot_info)
    
    #keep rows that have cells
    dots_in_cells = locations.loc[dot_info[:,0]]
    
    #add cell info
    dots_in_cells["cell number"] = dot_info[:,1]
    
    dots_in_cells.to_csv(str(output_path))
    
    print(str(output_path))

def keep_dots_parallel(mask,dot_locations):
    """run keep_dots_in_cells across all positions
    
    Parameters:
    -----------
    mask = list of numerically organized mask path
    dot_locations = list of numerically organized genes decoded paths
    
    """

    import time
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=12) as exe:
        futures = {}
        for i in range(len(mask)):
            fut = exe.submit(keep_dots_in_cells, mask[i], dot_locations[i])
            futures[fut] = i
        
        for fut in as_completed(futures):
            path = futures[fut]
            print(f'Path {path} completed after {time.time() - start} seconds')