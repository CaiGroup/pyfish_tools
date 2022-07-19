"""
authors: Katsuya Lex Colon and Lincoln Ombelets
group: Cai Lab
updated: 07/19/22
"""
#basic analysis package
import numpy as np
import pandas as pd
import time
#image analysis packages
from util import pil_imread
import tifffile as tf
import cv2
import sklearn.neighbors as nbrs
from photutils.detection import DAOStarFinder
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#organization packages
from pathlib import Path
import os
#fitting
from scipy.stats import norm
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

def dot_displacement(dist_arr, show_plot=True):
    """
    This function will calculate the localization precision by fitting a 1d gaussian
    to a distance array obtained from colocalizing dots. The full width half maximum of this
    1D gaussian will correspond to displacement.
    
    Parameters
    ----------
    dist_arr: 1D distance array
    output_folder: path to output error and plots
    
    Return
    ----------
    displacement
    """

    #create positive and negative distance array
    dist_arr = np.concatenate([-dist_arr,dist_arr])
    
    #fit gaussian distribution
    mu, std = norm.fit(dist_arr) 
    xmin, xmax = min(dist_arr), max(dist_arr)
    x = np.linspace(xmin, xmax, 500)
    p = norm.pdf(x, mu, std)
    
    #get half maximum of gaussian
    half_max = max(p)/2
    
    #get half width at half maximum
    index_hwhm = np.where(p > max(p)/2)[0][-1]
    
    #get displacement by looking at fullwidth
    displacement = x[index_hwhm]*2
    
    if show_plot == True:
        #plot histogram
        plt.hist(dist_arr, density=True, bins=50)
        #plot distribution
        plt.plot(x,p, label="Gaussian Fitted Data")
        #plot half max
        plt.axhline(half_max, color="red")
        #plot full width
        plt.axvline(displacement/2, color="red", label="FWHM")
        plt.axvline(-displacement/2, color="red")
        plt.legend()
        sns.despine()
        plt.ylabel("Probability density")
        plt.xlabel("Relative distances (pixels)")
        plt.show()
        plt.clf()
    
    return displacement

def get_optimum_fwhm(data, threshold,fwhm_range=(3,7)):
    """
    Finds the best fwhm
    Parameters
    ----------
    data = 2D array
    threshold = initial threshold for testing
    """
    #generate fwhm to test
    fwhm_range = np.linspace(fwhm_range[0],fwhm_range[1],4)
    #get counts
    counts = []
    for fwhm in fwhm_range:
        try:
            dots = len(daofinder(data,  threshold, fwhm))
        except:
            continue
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

def get_alignment_dots(image, threshold=100, fwhm_range=(5,10)):
    """
    This funtion will pick spots using daostarfinder.
    
    Parameters
    ----------
    image: image tiff
    threshold: pixel value the image must be above
    fwhm_range: range of fwhm to test
    
    Returns
    -------
    centroids
    """
    #get best fwhm
    fwhm = get_optimum_fwhm(image, threshold=threshold, fwhm_range=fwhm_range)
    
    #detect fiducials
    dots = daofinder(image, threshold=threshold, fwhm=fwhm)
    xy = dots[["xcentroid","ycentroid"]].to_list()
    
    return xy
    

def nearest_neighbors(ref_points, fit_points, max_dist=None):
    """
    This function finds corresponding points between two point sets of the same length using 
    nearest neighbors, optionally throwing away pairs that are above max_dist pixels apart.
    
    Parameters
    ----------
    ref_points = list containing x,y coord acting as reference 
    fit_points = list containing x,y coord for the dots we want to align to ref
    max_dist = number of allowed pixels two dots can be from each other
    
    Returns
    -------
    dists and a list of indices of fit_points which correspond.
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
    
    return dists, ref_indices, fit_indices
    
def nearest_neighbors_transform(ref_points, fit_points, max_dist=None, ransac_threshold = 0.5):
    """
    This function will take two lists of non-corresponding points and identify corresponding points less than max_dist apart
    using nearest_neighbors(). Then it will find a transform that wil bring the second set of dots to the first.
    Affine transformation with RANSAC was used to estimate transform. 
    
    Parameters
    ----------
    ref_points = list of x,y coord of ref
    fit_points = list of x,y coord of raw
    ransac_threshold = allowed error in ransac between ref and reprojected points to be considered inliers
    max_dist = maximum allowed distance apart two points can be in neighbor search
    
    Returns
    -------
    transform object, distances
    """
    
    #convert lists to arrays
    ref_points = np.array(ref_points)
    fit_points = np.array(fit_points)
    
    #check if dots have nan, if so remove
    ref_points = ref_points[~np.isnan(ref_points).any(axis=1)]
    fit_points = fit_points[~np.isnan(fit_points).any(axis=1)]
    
    #find nearest neighbors
    dists, ref_inds, fit_inds = nearest_neighbors(ref_points, fit_points, max_dist=max_dist)
    
    #get ref point coord and fit point coord used
    ref_pts_corr = ref_points[ref_inds]
    fit_pts_corr = fit_points[fit_inds]
    
    #estimate affine matrix using RANSAC
    tform = cv2.estimateAffine2D(fit_pts_corr, ref_pts_corr, ransacReprojThreshold=ransac_threshold)[0]

    return tform, dists, ref_pts_corr, fit_pts_corr

def alignment_error(ref_points_affine, moving_points_affine, 
                    ori_dist_list, tform_list, max_dist=2):
    
    """
    This function will calculate localization precision by obtaining FWHM of corrected distance array.

    Parameters
    ----------
    ref_points_affine = reference points used in transform
    moving_points_affine = points that are moving to reference
    ori_dist_list = original distance calculated prior to transform
    tform_list = list of affine transform matrix
    max_dist = number of allowed pixels two dots can be from each other

    Returns
    -------
    percent improved in alignment as well as alignment error
    """
    

    #apply transform to each moving point and calculate displacement 
    new_dist_by_channel = []
    old_dist_by_channel = []
    for i in range(len(moving_points_affine)):
        #reformat points
        moving = moving_points_affine[i].reshape(1, moving_points_affine[i].shape[0], moving_points_affine[i].shape[1])
        #perform transform on 2 coord points
        tform_points = cv2.transform(moving, tform_list[i])[0]
        
        #get new distance
        dists_new, _, _ = nearest_neighbors(ref_points_affine[i], tform_points, max_dist=max_dist)
        
        #remove distances beyond 2 pixels as they are most likely outliers after transform
        dists_new = dists_new[dists_new <= 2]
        
        #calculate localization precision
        displacement_new = dot_displacement(dists_new)
        displacement_old = dot_displacement(ori_dist_list[i], show_plot=False)
        new_dist_by_channel.append(displacement_new)
        old_dist_by_channel.append(displacement_old)
    
    #calculate percent improvement and average distance off
    percent_improvement_list = []
    for i in range(len(new_dist_by_channel)):
        percent_change = ((new_dist_by_channel[i]-old_dist_by_channel[i])/old_dist_by_channel[i])
        if percent_change < 0:
            percent_improvement = np.abs(percent_change)
            percent_improvement_list.append([i,percent_improvement,new_dist_by_channel[i]])
        else:
            percent_improvement = -percent_change
            percent_improvement_list.append([i,percent_improvement,new_dist_by_channel[i]])
            
    return percent_improvement_list
    
def chromatic_corr_offsets(tiff_src,threshold_abs=500, max_dist=2,
                           ransac_threshold = 0.5,
                           swapaxes=False):
    """
    This function will correct for chromatic aberration.
    
    Parameters
    ----------
    tiff_src = raw tiff source
    threshold_abs = absolute threshold for intensity
    max_dist = max distance for neighbor search
    ransac_threshold = allowed error in ransac between ref and reprojected points to be considered inliers
    swapaxes = bool to switch channel and z axes
    
    Returns
    -------
    Affine transformed image and alignment error
    """
    
    #read in image
    if swapaxes == True:
        tiff = pil_imread(tiff_src, swapaxes=True)
    else:
        tiff = pil_imread(tiff_src)
    
    #Get dots per channel 
    exp_dots_list = []
    #check if this is a single z
    if len(tiff.shape) == 3:
        number_of_channels = tiff.shape[0]-1
        for c in range(number_of_channels):
            #get dots from each channel
            exp_dots = get_alignment_dots(tiff[c], threshold=threshold_abs)
            exp_dots_list.append(exp_dots) 
    else:
        number_of_channels = tiff.shape[1]-1
        for c in range(number_of_channels):
            #max project image
            tiff_max = np.max(tiff[:,c,:,:], axis=0)
            #get alignment dots per channel
            exp_dots = get_alignment_dots(tiff_max,threshold_abs=threshold_abs)
            exp_dots_list.append(exp_dots)
            
    #get affine transform matrix (RANSAC) for each channel referenced to first channel or channel 488
    tform_list = []
    ori_dist_list = []
    ref_points_used= []
    fit_points_used= []
    
    for i in np.arange(1,len(exp_dots_list),1):
        tform, ori_dist, ref_pts, fit_pts = nearest_neighbors_transform(exp_dots_list[0], 
                                                                        exp_dots_list[i], 
                                                                        max_dist=max_dist, 
                                                                        ransac_threshold=ransac_threshold)
        ori_dist_list.append(ori_dist)
        tform_list.append(tform)
        ref_points_used.append(ref_pts)
        fit_points_used.append(fit_pts)
   
    #apply tform to each channel and across z 
    corr_stack = []

    for i in np.arange(1,len(tform_list)+1,1):
        #check if it is one z
        if len(tiff.shape) == 3:
            #add reference image
            if i == 1:
                corr_stack.append(tiff[0])
            corr_image = cv2.warpAffine(tiff[i],tform_list[i-1],dsize=(tiff[i].shape[0],tiff[i].shape[1]))
            corr_stack.append(corr_image)
            #tform list length should be minus dapi 
            #so we do -2 here to get the last i in loop
            #if we approach the last loop then we add unshifted dapi
            if i == (tiff.shape[0]-2):
                corr_image = tiff[i+1]
                corr_stack.append(corr_image)
        else:
            z_stack = []
            #add reference image
            if i == 1:
                for z in range(tiff.shape[0]):
                    z_stack.append(tiff[z][0])
                corr_stack.append(z_stack)
                z_stack = []
            for z in range(tiff.shape[0]):
                corr_image = cv2.warpAffine(tiff[z][i],tform_list[i-1],dsize=(tiff[z][i].shape[0],tiff[z][i].shape[1]))
                z_stack.append(corr_image)
            corr_stack.append(z_stack)
            #tform list length should be minus dapi 
            #so we do -2 here to get the last i in loop
            #if we approach the last loop then we add unshifted dapi
            if i == (tiff.shape[1]-2):
                z_stack = []
                for z in range(tiff.shape[0]):
                    corr_image = tiff[z][i+1]
                    z_stack.append(corr_image)
                corr_stack.append(z_stack)
   
    if len(tiff.shape) != 3:
        #switch axes to z,c,x,y
        transformed_image = np.swapaxes(np.array(corr_stack),0,1)
    else:
        transformed_image = np.array(corr_stack)
    
    #check alignment error
    error = alignment_error(ref_points_used, fit_points_used, ori_dist_list, tform_list, max_dist)
    
    #write error
    txt_name = Path(tiff_src).name.replace(".ome.tif","_error.txt")
    output_text =  Path(tiff_src).parent / txt_name
    with open(str(output_text),"w+") as f:
        for element in error:
            f.write(str(element[0]) + " " + str(element[1]) + " " + str(element[2]) + "\n")
    f.close()
    
    error = pd.DataFrame(error)
    error.columns = ["Channels (Reference Excluded)","Percent Improvement","FWHM"]
        
    return transformed_image, error, tform_list

def apply_tform(img_src, tform_list, swapaxes=False, write = True):
    
        """
        This function will apply the transformation matrix obtained from chromatic_corr_offsets() to an image.
        
        Parameters
        ----------
        img_src: path to image
        tform_list: list of transformation matricies
        write: bool to write image
        
        Returns
        -------
        corrected image
        """
        #output path
        parent = Path(img_src).parent
        while "notebook_pyfiles" not in os.listdir(parent):
            parent = parent.parent
        #create output path
        output_folder = parent / "notebook_pyfiles" / "aberration_corrected"
        hybcycle = Path(img_src).parent.name
        output_path = output_folder / hybcycle / Path(img_src).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        #read image
        tiff = pil_imread(img_src, swapaxes=swapaxes)

        #apply tform to each channel and across z 
        corr_stack = []
        for i in np.arange(1,len(tform_list)+1,1):
            #check if it is one z
            if len(tiff.shape) == 3:
                #add reference image
                if i == 1:
                    corr_stack.append(tiff[0])
                corr_image = cv2.warpAffine(tiff[i],tform_list[i-1],dsize=(tiff[i].shape[0],tiff[i].shape[1]))
                corr_stack.append(corr_image)
                #tform list length should be minus dapi 
                #so we do -2 here to get the last i in loop
                #if we approach the last loop then we add unshifted dapi
                if i == (tiff.shape[0]-2):
                    corr_image = tiff[i+1]
                    corr_stack.append(corr_image)
            else:
                z_stack = []
                #add reference image
                if i == 1:
                    for z in range(tiff.shape[0]):
                        z_stack.append(tiff[z][0])
                    corr_stack.append(z_stack)
                    z_stack = []
                for z in range(tiff.shape[0]):
                    corr_image = cv2.warpAffine(tiff[z][i],tform_list[i-1],dsize=(tiff[z][i].shape[0],tiff[z][i].shape[1]))
                    z_stack.append(corr_image)
                corr_stack.append(z_stack)
                #tform list length should be minus dapi 
                #so we do -2 here to get the last i in loop
                #if we approach the last loop then we add unshifted dapi
                if i == (tiff.shape[1]-2):
                    z_stack = []
                    for z in range(tiff.shape[0]):
                        corr_image = tiff[z][i+1]
                        z_stack.append(corr_image)
                    corr_stack.append(z_stack)
                        
        if len(tiff.shape) != 3:
            #switch axes to z,c,x,y
            transformed_image = np.swapaxes(np.array(corr_stack),0,1)
        else:
            transformed_image = np.array(corr_stack)
            
        if write == True:
            tf.imwrite(str(output_path),transformed_image)
        else:
            return transformed_image
            
def apply_chromatic_corr(tiff_srcs, tform_list, cores = 24, swapaxes=True, write = True):
    """
    This function will apply chromatic aberration correction of supplied list of images.
    
    Parameters 
    ----------
    tiff_srcs = list of image paths
    tform_list = list of affine transformation matrix
    cores = number of cores to use
    
    Returns
    -------
    outputs transformed images
    """
    
    #start time
    start = time.time()
    
    #check if it is only 1 image
    if type(tiff_srcs) != list:
        apply_tform(tiff_srcs, tform_list, swapaxes=swapaxes, write = write)
        print(f'Path {tiff_srcs} completed after {(time.time() - start)/60} minutes')
    else:
        with ProcessPoolExecutor(max_workers=cores) as exe:
            futures = {}
            for path in tiff_srcs:
                fut = exe.submit(apply_tform, path, tform_list, swapaxes=swapaxes, write=write)
                futures[fut] = path
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {(time.time() - start)/60} minutes')
