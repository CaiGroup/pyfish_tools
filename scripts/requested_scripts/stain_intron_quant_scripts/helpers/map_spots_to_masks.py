"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 07/24/22
"""

import numpy as np

def keep_dots_in_cells(img, locations, pos_info):
    """a function to remove any dots outside of mask
    Parameter
    ---------
    img: cellpose generated mask 
    dot_locations: dot locations
    pos_info: position number
    time_info : time course info
    
    Returns
    -------
    output locations.csv 
    
    """
    
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
            cell = f"cell{cell}_pos{pos_info}"
            dot_info.append([i,cell])
            
    dot_info = np.array(dot_info)
    
    #keep rows that have cells
    dots_in_cells = locations.iloc[dot_info[:,0]]
    
    #add cell info
    dots_in_cells["cell id"] = dot_info[:,1]
    
    return dots_in_cells
