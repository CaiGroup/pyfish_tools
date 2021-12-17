"""
author: Katsuya Lex Colon and Lincoln Ombelets
group: Cai Lab
updated: 12/03/21
"""
#basic analysis package
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile as tf
#image analysis packages
from skimage.feature import peak_local_max
from photutils.centroids import centroid_2dg
from scipy import ndimage
from skimage.transform import AffineTransform
import sklearn.neighbors as nbrs
from skimage import registration
#parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

def get_alignment_dots(image, region_size=10, min_distance=10, 
                       threshold_abs=500, num_peaks=1000):
    """
    This funtion will pick dots using skimage.feature.peak_local_max then generate a bounding box
    for that dot. The isolated dot will then be fitted with a 2d gaussian to get subpixel centers.
    
    Parameters
    ----------
    image = image tiff
    region_size = size of bounding box
    min_distance = minimum number of pixels separating peaks (arg for peal_local_max)
    threshold_abs = minimum absolute pixel intensity
    num_peaks = number of desired dots
    """
    
    #pick out bright dots
    dot_cands = peak_local_max(
        image, 
        min_distance=min_distance, 
        threshold_abs=threshold_abs, 
        num_peaks=num_peaks
    )
    
    centroids = []
    #gaussian fit dots to get subpixel centroids
    #cand should by y,x coord
    for cand in dot_cands:
        try:
            im_data = get_region_around(image, cand, region_size)
            x_g, y_g = centroid_2dg(im_data)
            y_offset = np.abs(y_g-(region_size / 2))
            x_offset = np.abs(x_g-(region_size / 2))
            #apply offset to dots
            if y_g > (region_size / 2):
                y = cand[0]+y_offset
            else:
                y = cand[0]-y_offset
            if x_g > (region_size / 2):
                x = cand[1]+x_offset
            else:
                x = cand[1]-x_offset
            centroids.append([x, y])
        except ValueError:
            continue
        except IndexError:
            continue
    
    return centroids


def nearest_neighbors(ref_points, fit_points, max_dist=None):
    """
    This function finds corresponding points between two point sets of the same length using 
    nearest neighbors, optionally throwing away pairs that are above max_dist pixels apart,
    and optionally aggregating the distance vector (e.g. for minimization of some norm)
    
    Parameters
    ----------
    ref_points = array containing x,y coord acting as reference 
    fit_points = array containing x,y coord for the dots we want to align to ref
    max_dist = number of allowed pixels two dots can be from each other
    
    Returns
    -------
    dists or agg(dists); a vector of indices of ref_points; and a vector of
    indices of fit_points which correspond.
    """
    #initiate neighbors
    ref_neighbors = nbrs.NearestNeighbors(n_neighbors=1).fit(ref_points)
    #perform search
    dists, ref_indices = ref_neighbors.kneighbors(fit_points)
    #get distance values for nearest neighbor
    dists = np.array(dists)[:, 0]
    #flatten indicies
    ref_indices = ref_indices.ravel()
    #generate indicies for fit
    fit_indices = np.arange(0, len(fit_points), 1)
    
    #remove pairs over a max dist
    if max_dist is not None:
        to_drop = np.where(dists > max_dist)
        
        dists[to_drop] = -1
        ref_indices[to_drop] = -1
        fit_indices[to_drop] = -1
                
        dists = np.compress(dists != -1, dists)
        ref_indices = np.compress(ref_indices != -1, ref_indices)
        fit_indices = np.compress(fit_indices != -1, fit_indices)
    
    ref_dots = np.array(ref_points)[ref_indices]
    fit_dots = np.array(fit_points)[fit_indices]
    
    return dists, ref_dots, fit_dots

def make_fiducial_mask(image_ref, image_moving, size=9,min_distance=10,threshold_abs=500,
                       num_peaks=1000, max_dist=2,normalize=True, edge='raise'):
    """
    This function will essentially get a bounding box around detected dots
    
    Parameters
    ----------
    im = image tiff
    center = x,y centers from dot detection
    size = size of bounding box
    normalize = bool to normalize intensity
    edge = "raise" will output error message if dot is at border and
            "return" will adjust bounding box 
            
    Returns
    -------
    array of boxed dot region
    """
    #pick out bright dots
    cands_ref = peak_local_max(image_ref, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks)
    cands_moving = peak_local_max(image_moving, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks)
    
    #find best matches
    dists, ref_dots, fit_dots = nearest_neighbors(cands_ref, cands_moving, max_dist=max_dist)
    
    #copy image
    mask_ref = image_ref.copy()
    mask_moving = image_moving.copy()
    
    for dots in ref_dots:
        #calculate bounds
        lower_bounds = np.array(dots).astype(int) - size//2
        upper_bounds = np.array(dots).astype(int) + size//2 + 1

        #check to see if bounds is on edge
        if any(lower_bounds < 0) or any(upper_bounds > image_ref.shape[-1]):
            if edge == 'raise':
                raise IndexError(f'Center {center} too close to edge to extract size {size} region')
            elif edge == 'return':
                lower_bounds = np.maximum(lower_bounds, 0)
                upper_bounds = np.minimum(upper_bounds, image_ref.shape[-1])

        #convert region into 1
        mask_ref[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]=True
        
    for dots in fit_dots:
        #calculate bounds
        lower_bounds = np.array(dots).astype(int) - size//2
        upper_bounds = np.array(dots).astype(int) + size//2 + 1

        #check to see if bounds is on edge
        if any(lower_bounds < 0) or any(upper_bounds > image_moving.shape[-1]):
            if edge == 'raise':
                raise IndexError(f'Center {center} too close to edge to extract size {size} region')
            elif edge == 'return':
                lower_bounds = np.maximum(lower_bounds, 0)
                upper_bounds = np.minimum(upper_bounds, image_moving.shape[-1])

        #convert region into 1
        mask_moving[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]=True

    #make boolean mask
    return (mask_ref == True, mask_moving == True)

# def phase_correlation():
#     #calculate shift on max projected dapi
#     shift,error = registration.phase_cross_correlation(
#         beads[0][0],raw[0][0], reference_mask = ref, 
#         moving_mask = moving, upsample_factor=20)