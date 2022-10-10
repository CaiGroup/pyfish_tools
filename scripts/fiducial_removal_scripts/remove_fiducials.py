"""
author: Katsuya Lex Colon
group: Cai Lab
date: 07/25/22
"""

#basic analysis package
import numpy as np
import pandas as pd
#image analysis packages
from photutils.detection import DAOStarFinder
from sklearn.neighbors import NearestNeighbors
from util import pil_imread
import cv2
#parallel processing
from concurrent.futures import ProcessPoolExecutor
#organization packages
from pathlib import Path
import os
#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def check_axis(img):
    """
    Determine if the img axis needs to be flipped if both channel and z axis is the same
    Parameters
    ----------
    img = numpy 4d array
    """
    #performing normalized correlation analysis on expected dapi channel
    ax1_list = []
    for z in np.arange(0, img.shape[0]-1, 1):
        ref_compressed = img[z][-1].astype(np.float32)
        src_compressed = img[z+1][-1].astype(np.float32)
        corr = cv2.matchTemplate(ref_compressed, src_compressed, cv2.TM_CCOEFF_NORMED)
        ax1_list.append(corr)
        
    ax2_list = []
    for z in np.arange(0, img.shape[1]-1, 1):
        ref_compressed = img[-1][z].astype(np.float32)
        src_compressed = img[-1][z+1].astype(np.float32)
        corr = cv2.matchTemplate(ref_compressed, src_compressed, cv2.TM_CCOEFF_NORMED)
        ax2_list.append(corr)
     
    #axis with highest correlation should be the correct shape    
    correct_axis = np.argmax([np.mean(ax1_list), np.mean(ax2_list)])
    
    if correct_axis == 1:
        img = np.swapaxes(img, 0, 1)
    
    return img

def get_optimum_fwhm(data, threshold):
    """
    Finds the best fwhm
    Parameters
    ----------
    data = 2D array
    threshold = initial threshold for testing
    """
    #generate fwhm to test
    fwhm_range = np.linspace(3,10,8)
    #get counts
    counts = []
    for fwhm in fwhm_range:
        dots = len(daofinder(data,  threshold, fwhm))
        counts.append(dots)
    #find index with largest counts
    best_index = np.argmax(counts)
    #this is the best fwhm
    best_fwhm = fwhm_range[best_index]
    
    return best_fwhm

def daofinder(data,  threshold, fwhm = 4.0):
    """
    This function will return the output of daostarfinder
    Parameters
    ----------
    data = 2D array
    threshold = absolute intensity threshold
    fwhm = full width half maximum
    """
    #use daostarfinder to pick dots
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=None, exclude_border=True)
    sources = daofind(data)
    
    #return none if nothing was picked else return table
    if sources == None:
        return None
    for col in sources.colnames:
         sources[col].info.format = '%.8g'  # for consistent table output
    
    return sources.to_pandas()

def find_fiducials(image, threshold=100):
    """
    This funtion will pick dots using skimage.feature.peak_local_max then generate a bounding box
    for that dot. The isolated dot will then be fitted with a 2d gaussian to get subpixel centers.
    
    Parameters
    ----------
    image = image tiff
    region_size = size of bounding box (use odd number)
    min_distance = minimum number of pixels separating peaks (arg for peal_local_max)
    threshold_abs = minimum absolute pixel intensity
    num_peaks = number of desired dots
    
    Returns
    -------
    centroids
    """
    #get best fwhm
    fwhm = get_optimum_fwhm(image, threshold=threshold)
    
    #detect fiducials
    dots = daofinder(image, threshold=threshold, fwhm=fwhm)
    xy = dots[["xcentroid","ycentroid","flux", "peak"]]
    xy.columns = ["x","y", "flux", "peak intensity"]
    
    return xy
        
def remove_fiducials(fiducials, locations, radius=1):
    """
    Remove dots that overlap with fiducial locations using nearest neighbor searches.
    
    Parameters
    ----------
    fiducials = locations of fiducials
    locations = locations of real dots
    radius = search radius
    """
    
    #reset index for df just in case
    fiducials = fiducials.reset_index(drop=True)
    locations = locations.reset_index(drop=True)
    
    #using sklearn nearest neighbor algorithm to find nearest dots
    #initialize algorithm
    neigh = NearestNeighbors(n_neighbors=2, radius=radius, metric="euclidean", n_jobs=1)
    
    #initialize neighbor
    initial_seed = fiducials[["x","y"]]
    #find neighbors for fiducials
    neigh.fit(locations[["x","y"]])
    distances,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
    
    #nearest neighbor dot
    neighbors_flattened = []
    for i in range(len(neighbors)):
        try:
            neighbors_flattened.append([i,neighbors[i][0]])
        except IndexError:
            continue
            
    #separate file for dots that do not colocalize
    locations_nocoloc_idx = np.array(list(set(locations.index)-set(np.array(neighbors_flattened)[:,1])))
    locations_nocoloc = locations.iloc[locations_nocoloc_idx].reset_index(drop=True)
        
    return locations_nocoloc

def remove_all_fiducials(locations_src, fid_src, threshold=500, radius=1, num_channels=4, write =True):
    """
    Remove fiducials for each z and channel for a specific position.
    
    Parameters
    ----------
    locations_src: file path for dot locations
    fid_src: file path for fiducial images
    threshold: absolute pixel intensity threshold fiducials must surpass
    radius: pixel search
    num_channels: number of channels in your image
    write: bool to write results
    """
    #swapaxes to z,c,x,y if required
 
    fiducials = pil_imread(fid_src, swapaxes=True)
    if fiducials.shape[1] != num_channels:
        fiducials = pil_imread(fid_src, swapaxes=False)
        if fiducials.shape[0] == fiducials.shape[1]:
            fiducials = check_axis(fiducials)
    
    #read in locations file
    locations = pd.read_csv(locations_src)
    
    #check if there is z, else reshape
    if len(fiducials.shape) == 3:
        fiducials = fiducials.reshape(1, fiducials.shape[0], fiducials.shape[1], fiducials.shape[2])
     
    #go through each z and channel of locations and remove fiducial dots
    removed_locations = []
    fiducial_locations = []
    for z in locations["z"].unique().astype(int):
        for c in locations["ch"].unique().astype(int):
            fid_loc = find_fiducials(fiducials[z, c-1], threshold=threshold)
            fid_loc["z"]=z
            fid_loc["ch"]=c
            fiducial_locations.append(fid_loc)
            #go through each hyb, getting matching z and channel
            for hyb in locations["hyb"].unique():
                locations_slice = locations[(locations["hyb"]==hyb) & (locations["ch"]==c) &
                                            (locations["z"] == z)]
                
                fiducials_removed = remove_fiducials(fid_loc, locations_slice, radius=radius)
                removed_locations.append(fiducials_removed)
    #combine final
    final = pd.concat(removed_locations).reset_index(drop=True)
    fid_final = pd.concat(fiducial_locations).reset_index(drop=True)
    
    if write == False:
        return final, fid_final
    else:
        locations_parent = Path(locations_src).parent
        pos = locations_parent.name
        while "seqFISH_datapipeline" not in os.listdir(locations_parent):
            locations_parent = locations_parent.parent
        
        for ch in final["ch"].unique().astype(int):
            for z in final["z"].unique().astype(int):
                output_dir = locations_parent /"seqFISH_datapipeline"/ "output" /"dots_detected"/ "fiducials_removed" /f"Channel_{ch}" /pos
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"locations_z_{z}.csv"
                final_slice = final[(final["z"]==z) & (final["ch"]==ch)]
                final_slice.reset_index(drop=True).to_csv(str(output_path))

def remove_fiducials_parallel(locations_srcs, fid_srcs, threshold=500, radius=1, num_channels=4):
    """
    Remove fiducials from multiple locations files.
    
    Parameters
    ----------
    locations_srcs: list of file path for dot locations
    fid_srcs: list of file path for fiducial images
    threshold: absolute pixel intensity threshold fiducials must surpass
    radius: pixel search
    num_channels: number of total channels in image
    """
    
    if len(locations_srcs) == 1:
        remove_all_fiducials(locations_srcs, fid_srcs, threshold=threshold, radius=radius, num_channels=num_channels, write =True)
    
    else:
        with ProcessPoolExecutor(max_workers=20) as exe:
            for i in range(len(locations_srcs)):
                exe.submit(remove_all_fiducials, locations_srcs[i], fid_srcs[i],
                           threshold=threshold, radius=radius, num_channels=num_channels, write =True )
                print(f"fid_src={fid_srcs[i]}; dot_src={locations_srcs[i]}")