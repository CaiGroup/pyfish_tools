"""
authors: Katsuya Lex Colon and Lincoln Ombelets
group: Cai Lab
updated: 03/02/22
"""
#basic analysis package
import numpy as np
import pandas as pd
import time
#image analysis packages
from skimage.feature import peak_local_max
from photutils.centroids import centroid_2dg
from scipy import ndimage
from util import pil_imread
import tifffile as tf
import cv2
import sklearn.neighbors as nbrs
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#organization packages
from pathlib import Path
import glob
#fitting
from scipy.stats import norm
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

def get_region_around(im, center, size, normalize=True, edge='raise'):
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
    
    #calculate bounds
    lower_bounds = np.array(center) - size//2
    upper_bounds = np.array(center) + size//2 + 1
    
    #check to see if bounds is on edge
    if any(lower_bounds < 0) or any(upper_bounds > im.shape[-1]):
        if edge == 'raise':
            raise IndexError(f'Center {center} too close to edge to extract size {size} region')
        elif edge == 'return':
            lower_bounds = np.maximum(lower_bounds, 0)
            upper_bounds = np.minimum(upper_bounds, im.shape[-1])
    
    #slice out array of interest
    region = im[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]
    
    return region

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
    
def get_alignment_dots(image, region_size=7, min_distance=10, 
                       threshold_abs=500, num_peaks=1000):
    """
    This funtion will pick dots using skimage.feature.peak_local_max then generate a bounding box
    for that dot. The isolated dot will then be fitted with a 2d gaussian to get subpixel centers.
    
    Parameters
    ----------
    image = image to pick dots
    region_size = size of bounding box (use odd number)
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
   
    #gaussian fit dots to get subpixel centroids
    #cand should by y,x coord
    centroids = []
    for cand in dot_cands:
        try:
            im_data = get_region_around(image, cand, region_size)
            x_g, y_g = centroid_2dg(im_data)
            y_offset = np.abs(y_g-(region_size // 2))
            x_offset = np.abs(x_g-(region_size // 2))
            #apply offset to dots
            if y_g > (region_size // 2):
                y = cand[0]+y_offset
            else:
                y = cand[0]-y_offset
            if x_g > (region_size // 2):
                x = cand[1]+x_offset
            else:
                x = cand[1]-x_offset
            if (x > 0) and (y > 0):
                centroids.append([x, y])
            #could get nan by random chance if detecting giant blobs
            else:
                continue
        except ValueError:
            continue
        except IndexError:
            continue
            
    return centroids

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
    
def chromatic_corr_offsets(tiff_src,region_size=7, min_distance=10, 
                           threshold_abs=500, num_peaks=1000, max_dist=2,
                           ransac_threshold = 0.5, use_488 = False,
                           swapaxes=False):
    """
    This function will correct for chromatic aberration.
    
    Parameters
    ----------
    tiff_src = raw tiff source
    region_size = the bounding box size (best to use odd number)
    min_distance = number of min pixels two peaks should be apart
    threshold_abs = absolute threshold for intensity
    num_peaks = number of dots detected
    max_dist = max distance for neighbor search
    ransac_threshold = allowed error in ransac between ref and reprojected points to be considered inliers
    use_488 = bool to use 488 channel as reference
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
            exp_dots = get_alignment_dots(tiff[c], region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks)
            exp_dots_list.append(exp_dots) 
    else:
        number_of_channels = tiff.shape[1]-1
        for c in range(number_of_channels):
            #max project image
            tiff_max = np.max(tiff[:,c,:,:], axis=0)
            #get alignment dots per channel
            exp_dots = get_alignment_dots(tiff_max, region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks)
            exp_dots_list.append(exp_dots)
            
    #get affine transform matrix (RANSAC) for each channel referenced to first channel or channel 488
    tform_list = []
    ori_dist_list = []
    ref_points_used= []
    fit_points_used= []
    if use_488 == False:
        for i in np.arange(1,len(exp_dots_list),1):
            tform, ori_dist, ref_pts, fit_pts = nearest_neighbors_transform(exp_dots_list[0], 
                                                                            exp_dots_list[i], 
                                                                            max_dist=max_dist, 
                                                                            ransac_threshold=ransac_threshold)
            ori_dist_list.append(ori_dist)
            tform_list.append(tform)
            ref_points_used.append(ref_pts)
            fit_points_used.append(fit_pts)
    else:
        #get 488 nm slice and align
        channel_ref = number_of_channels-1
        for i in np.arange(0,len(exp_dots_list)-1,1):
            tform, ori_dist, ref_pts, fit_pts = nearest_neighbors_transform(exp_dots_list[channel_ref], 
                                                                            exp_dots_list[i], 
                                                                            max_dist=max_dist, 
                                                                            ransac_threshold=ransac_threshold)
            ori_dist_list.append(ori_dist)
            tform_list.append(tform)
            ref_points_used.append(ref_pts)
            fit_points_used.append(fit_pts)
        
    #apply tform to each channel and across z 
    corr_stack = []
    if use_488 == False:
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
    else:
        for i in np.arange(0,len(tform_list),1):
            #check if it is one z
            if len(tiff.shape) == 3:
                corr_image = cv2.warpAffine(tiff[i],tform_list[i],dsize=(tiff[i].shape[0],tiff[i].shape[1]))
                corr_stack.append(corr_image)
                #tform list length should be minus dapi and 488
                #so we do -1 here to get the last i in loop
                #if we approach the last loop then we add 488 ref and unshifted dapi 
                if i == (len(tform_list)-1):
                    #add reference image
                    corr_stack.append(tiff[channel_ref])
                    corr_image = tiff[i+2]
                    corr_stack.append(corr_image)
            else:
                z_stack = []
                for z in range(tiff.shape[0]):
                    corr_image = cv2.warpAffine(tiff[z][i],tform_list[i],dsize=(tiff[z][i].shape[0],tiff[z][i].shape[1]))
                    z_stack.append(corr_image)
                corr_stack.append(z_stack)
                #tform list length should be minus dapi and 488
                #so we do -1 here to get the last i in loop
                #if we approach the last loop then we add 488 ref and unshifted dapi 
                if i == (len(tform_list)-1):
                    z_stack_dapi = []
                    z_stack_ref = []
                    for z in range(tiff.shape[0]):
                        z_stack_ref.append(tiff[z][channel_ref])
                        corr_image = tiff[z][i+2]
                        z_stack_dapi.append(corr_image)
                    corr_stack.append(z_stack_ref)
                    corr_stack.append(z_stack_dapi)
    
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
        
    return transformed_image, error, tform_list

def apply_tform(img_src, tform_list, use_488=False, swapaxes=False, write = True):
    
        """
        This function will apply the transformation matrix obtained from chromatic_corr_offsets() to an image.
        
        Parameters
        ----------
        img_src: path to image
        tform_list: list of transformation matricies
        write: bool to write image
        use_488 = bool to use 488 channel as reference
        
        Returns
        -------
        corrected image
        """
        
        #create output path
        orig_image_dir = Path(img_src).parent.parent.parent
        output_folder = Path(orig_image_dir) / "aberration_corrected"
        hybcycle = Path(img_src).parent.name
        output_path = output_folder / hybcycle / Path(img_src).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        #read image
        tiff = pil_imread(img_src, swapaxes=swapaxes)

        #apply tform to each channel and across z 
        corr_stack = []
        if use_488 == False:
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
        else:
            for i in np.arange(0,len(tform_list),1):
                #check if it is one z
                if len(tiff.shape) == 3:
                    corr_image = cv2.warpAffine(tiff[i],tform_list[i],dsize=(tiff[i].shape[0],tiff[i].shape[1]))
                    corr_stack.append(corr_image)
                    #tform list length should be minus dapi and 488
                    #so we do -1 here to get the last i in loop
                    #if we approach the last loop then we add 488 ref and unshifted dapi 
                    if i == (len(tform_list)-1):
                        #add reference image
                        corr_stack.append(tiff[channel_ref])
                        corr_image = tiff[i+2]
                        corr_stack.append(corr_image)
                else:
                    z_stack = []
                    for z in range(tiff.shape[0]):
                        corr_image = cv2.warpAffine(tiff[z][i],tform_list[i],dsize=(tiff[z][i].shape[0],tiff[z][i].shape[1]))
                        z_stack.append(corr_image)
                    corr_stack.append(z_stack)
                    #tform list length should be minus dapi and 488
                    #so we do -1 here to get the last i in loop
                    #if we approach the last loop then we add 488 ref and unshifted dapi 
                    if i == (len(tform_list)-1):
                        z_stack_dapi = []
                        z_stack_ref = []
                        for z in range(tiff.shape[0]):
                            z_stack_ref.append(tiff[z][channel_ref])
                            corr_image = tiff[z][i+2]
                            z_stack_dapi.append(corr_image)
                        corr_stack.append(z_stack_ref)
                        corr_stack.append(z_stack_dapi)
                        
        if len(tiff.shape) != 3:
            #switch axes to z,c,x,y
            transformed_image = np.swapaxes(np.array(corr_stack),0,1)
        else:
            transformed_image = np.array(corr_stack)
            
        if write == True:
            tf.imwrite(str(output_path),transformed_image)
        else:
            return transformed_image
            
def apply_chromatic_corr(tiff_srcs, tform_list, cores = 24, use_488=False, swapaxes=True, write = True):
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
        apply_tform(tiff_srcs, tform_list, use_488=use_488, swapaxes=swapaxes, write = write)
        print(f'Path {tiff_srcs} completed after {(time.time() - start)/60} minutes')
    else:
        with ProcessPoolExecutor(max_workers=cores) as exe:
            futures = {}
            for path in tiff_srcs:
                fut = exe.submit(apply_tform, path, tform_list, use_488=use_488, swapaxes=swapaxes, write=write)
                futures[fut] = path
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {(time.time() - start)/60} minutes')
