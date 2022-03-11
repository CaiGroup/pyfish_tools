"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 03/03/22
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
#fitting
from scipy.stats import norm
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#organization packages
from pathlib import Path
import glob
#ignore warnings
import warnings
warnings.filterwarnings("ignore")
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

def get_region_around(im, center, size, edge='raise'):
    """
    This function will essentially get a bounding box around detected dots
    
    Parameters
    ----------
    im = image tiff
    center = x,y centers from dot detection
    size = size of bounding box
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
        #calculate number of bins based on sturge's rule
        bins = round(1 + 3.322*np.log(len(dist_arr)))
        #plot histogram
        plt.hist(dist_arr, density=True, bins=bins)
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

def centroid_offset(x, y, x_g, y_g, region_size):
    """
    Returns corrected centroids
    Parameters
    ----------
    x = uncorrected x coord
    y = uncorrected y coord
    x_g = gaussian fitted x 
    y_g = gaussian fitted y
    region_size = size of bounding box
    """
    #calculate offset
    y_offset = np.abs(y_g-(region_size // 2))
    x_offset = np.abs(x_g-(region_size // 2))

    #apply offset to dots
    if y_g > (region_size // 2):
        y = y+y_offset
    else:
        y = y-y_offset
    if x_g > (region_size // 2):
        x = x+x_offset
    else:
        x = x-x_offset
    
    return x,y
    
def get_alignment_dots(image, ref_coord = None, region_size=7, min_distance=10, 
                       threshold_abs=500, num_peaks=1000, use_ref_coord=False):
    """
    This funtion will pick dots using skimage.feature.peak_local_max then generate a bounding box
    for that dot. The isolated dot will then be fitted with a 2d gaussian  or radial centered to get subpixel centers.
    
    Parameters
    ----------
    image = image tiff
    ref_coord = reference coordinates
    region_size = size of bounding box (use odd number)
    min_distance = minimum number of pixels separating peaks (arg for peal_local_max)
    threshold_abs = minimum absolute pixel intensity
    num_peaks = number of desired dots
    use_ref_coord = bool to use reference coordinates to locate dots in moving image
    
    Returns
    -------
    centroids
    """
    if use_ref_coord == False:
        #pick out bright dots
        dot_cands = peak_local_max(
            image, 
            min_distance=min_distance, 
            threshold_abs=threshold_abs, 
            num_peaks=num_peaks
        )
    else:
        #convert reference coordinates to int
        dot_cands = np.array(ref_coord).astype(int)
        #swap columns to make it y,x
        dot_cands = dot_cands[:,[1, 0]]

    #gaussian fit dots to get subpixel centroids
    #cand should by y,x coord
    centroids = []
    for cand in dot_cands:
        try:
            #get initial fit
            im_data = get_region_around(image, cand, region_size)
            x_g, y_g = centroid_2dg(im_data)
            
            #get corrected coordinates
            x,y = centroid_offset(cand[1], cand[0], x_g, y_g, region_size)
                
            #generate adjusted bounding box and fit again
            #this step will be important if the fiducial is off by 1 pixel
            im_data = get_region_around(image, [int(y), int(x)], region_size)
            x_g, y_g = centroid_2dg(im_data)
            
            #get corrected coordinates
            x_fin,y_fin = centroid_offset(int(x), int(y), x_g, y_g, region_size)
            
            centroids.append([x_fin, y_fin])
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
    max_dist = maximum allowed distance apart two points can be in neighbor search
    ransac_threshold = adjust the max allowed error in pixels
    
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
    
    #calculate percent improvement
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
    
def fiducial_alignment_single(tiff_src, ref_src,region_size=7, min_distance=10, 
                              threshold_abs=500, num_peaks=1000, max_dist=2,ransac_threshold=0.5,
                              include_dapi=True, use_ref_coord=False, swapaxes=False, write = True):
    """
    Parameters
    ----------
    tiff_src = raw tiff source
    ref_src = the reference image to align the image
    region_size = the bounding box size (best to use odd number)
    min_distance = number of min pixels two peaks should be apart
    threshold_abs = absolute threshold for intensity
    num_peaks = number of dots detected
    max_dist = max distance for neighbor search
    ransac_threshold = adjust the max allowed error in pixels
    include_dapi = bool to include dapi for alignment
    use_ref_coord = bool to use reference coordinates to locate dots in moving image
    swapaxes = bool to switch channel and z axes
    write = bool to write image
    
    Returns
    -------
    Affine transformed image and alignment error
    """
    #create output path
    orig_image_dir = Path(tiff_src).parent.parent
    output_folder = Path(orig_image_dir) / "fiducial_aligned"
    hybcycle = Path(tiff_src).parent.name
    output_path = output_folder / hybcycle / Path(tiff_src).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read in image
    if swapaxes == True:
        tiff = pil_imread(tiff_src, swapaxes=True)
        ref = pil_imread(ref_src, swapaxes=True)
    else:
        tiff = pil_imread(tiff_src)
        ref = pil_imread(ref_src)
    
    #Get dots per channel 
    ref_dots_list = []
    exp_dots_list = []
    #check if this is a single z
    if len(tiff.shape) == 3:
        if include_dapi == True:
            number_of_channels = tiff.shape[0]
        else: 
            number_of_channels = tiff.shape[0]-1
        for c in range(number_of_channels):
            #get reference and experimental alignment dots
            ref_dots = get_alignment_dots(ref[c], ref_coord=None, region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=False)
            if use_ref_coord == False:
                exp_dots = get_alignment_dots(tiff[c], ref_coord=None, region_size=region_size, min_distance=min_distance, 
                               threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=False)
            else:
                exp_dots = get_alignment_dots(tiff[c], ref_coord=ref_dots, region_size=region_size, min_distance=min_distance, 
                               threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=True)
                
            ref_dots_list.append(ref_dots)
            exp_dots_list.append(exp_dots) 
    else:
        if include_dapi == True:
            number_of_channels = tiff.shape[1]
        else:
            number_of_channels = tiff.shape[1]-1
        for c in range(number_of_channels):
            #max project image
            ref_max = np.max(ref[:,c,:,:], axis=0)
            tiff_max = np.max(tiff[:,c,:,:], axis=0)
            #get alignment dots per channel
            ref_dots = get_alignment_dots(ref_max, ref_coord=None, region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=False)
            if use_ref_coord == False:
                exp_dots = get_alignment_dots(tiff_max, ref_coord=None, region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=False)
            else:
                exp_dots = get_alignment_dots(tiff_max, ref_coord=ref_dots, region_size=region_size, min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks, use_ref_coord=True)
            ref_dots_list.append(ref_dots)
            exp_dots_list.append(exp_dots)
            
    #get affine transform matrix, original distance, reference points and moving points used for each channel
    tform_list = []
    ori_dist_list = []
    ref_points_used= []
    fit_points_used= []
    for i in range(len(ref_dots_list)):
        tform, ori_dist, ref_pts, fit_pts = nearest_neighbors_transform(ref_dots_list[i], exp_dots_list[i],
                                                               max_dist=max_dist,ransac_threshold=ransac_threshold)
        ori_dist_list.append(ori_dist)
        tform_list.append(tform)
        ref_points_used.append(ref_pts)
        fit_points_used.append(fit_pts)
        
    #apply tform to each channel and across z 
    corr_stack = []
    for i in range(len(tform_list)):
        #check if it is one z
        if len(tiff.shape) == 3:
            if include_dapi == True:
                corr_image = cv2.warpAffine(tiff[i],tform_list[i],dsize=(tiff[i].shape[1],tiff[i].shape[0]))
                corr_stack.append(corr_image)
            else:
                corr_image = cv2.warpAffine(tiff[i],tform_list[i],dsize=(tiff[i].shape[1],tiff[i].shape[0]))
                corr_stack.append(corr_image)
                #tform list length should be minus dapi if include_dapi == False
                #so we do -2 here to get the last i in loop
                #if we approach the last loop then we add unshifted dapi
                if i == (tiff.shape[0]-2):
                    corr_image = tiff[i+1]
                    corr_stack.append(corr_image)
        else:
            z_stack = []
            z_stack_dapi = []
            for z in range(tiff.shape[0]):
                if include_dapi == True:
                    corr_image = cv2.warpAffine(tiff[z][i],tform_list[i],dsize=(tiff[z][i].shape[1],tiff[z][i].shape[0]))
                    z_stack.append(corr_image)
                else:
                    corr_image = cv2.warpAffine(tiff[z][i],tform_list[i],dsize=(tiff[z][i].shape[1],tiff[z][i].shape[0]))
                    z_stack.append(corr_image)
                    #tform list length should be minus dapi if include_dapi == False
                    #so we do -2 here to get the last i in loop
                    #if we approach the last loop then we add unshifted dapi
                    if i == (tiff.shape[1]-2):
                        corr_image = tiff[z][i+1]
                        z_stack_dapi.append(corr_image)
            corr_stack.append(z_stack)
            if (include_dapi == False) and (i == (tiff.shape[1]-2)):
                corr_stack.append(z_stack_dapi)
    
    if len(tiff.shape) != 3:
       #switch axes to z,c,x,y
       transformed_image = np.swapaxes(np.array(corr_stack),0,1)
    
    #check alignment error
    error = alignment_error(ref_points_used, fit_points_used, ori_dist_list, tform_list, max_dist)
    
    if write == True:
        print(output_path)
        #write image
        tf.imwrite(str(output_path), transformed_image)
        #write error
        txt_name = Path(tiff_src).name.replace(".ome.tif","_error.txt")
        output_text =  output_folder / hybcycle / txt_name
        with open(str(output_text),"w+") as f:
            for element in error:
                f.write(str(element[0]) + " " + str(element[1]) + " " + str(element[2]) + "\n")
        f.close()
    else:    
        error = pd.DataFrame(error)
        error.columns = ["Channels","Percent Improvement","FWHM"]
        
        return transformed_image, error

def fiducial_align_parallel(tiff_list, ref_src, region_size=7, min_distance=10, 
                            threshold_abs=500, num_peaks=1000, max_dist=2,ransac_threshold=0.5,
                            include_dapi=True, use_ref_coord = False, swapaxes=False, cores = 24):
    """
    This function will run the fiducial alignment in parallel analyzing multiple images at once.
    
    Parameters 
    ----------
    tiff_list = list of tiff sources
    ref_src = path for reference image
    region_size = bounding box (use odd number)
    min_distance = minimum pixel distance between peaks for dot picking
    threshold_abs = absolute threshold value for peak detection
    num_peaks= number of desired dots
    max_dist = max distance for neighbor search
    ransac_threshold = adjust the max allowed error in pixels
    include_dapi = bool to include dapi channel
    use_ref_coord = bool to use reference coordinates to locate dots in moving image
    swapaxes = bool to switch z and c axes
    cores = number of cores to use
    
    Returns
    -------
    outputs transformed images and error
    """
    #start time
    start = time.time()
    
    #check if it is only 1 image
    if type(tiff_list) != list:
        fiducial_alignment_single(tiff_list, ref_src,region_size=region_size, min_distance=min_distance,
                                  threshold_abs=threshold_abs, num_peaks=num_peaks, max_dist=max_dist, 
                                  ransac_threshold=ransac_threshold,
                                  include_dapi=include_dapi, use_ref_coord= use_ref_coord, swapaxes=swapaxes, write = True)
        print(f'Path {tiff_list} completed after {(time.time() - start)/60} minutes')
    else:
        with ProcessPoolExecutor(max_workers=cores) as exe:
            futures = {}
            for path in tiff_list:
                fut = exe.submit(fiducial_alignment_single, path, ref_src, region_size=region_size, 
                                 min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks,
                                 ransac_threshold=ransac_threshold,
                                 max_dist=max_dist,include_dapi=include_dapi, use_ref_coord= use_ref_coord,
                                 swapaxes=swapaxes, write = True)
                futures[fut] = path
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {(time.time() - start)/60} minutes')

def plot_error(path_to_files, num_hybcycles = 80, num_channels = 4, savefig = True, by_pos = False):
    """
    This function will plot the fiducial alignment error and percent improvement
    Parameters
    ----------
    path_to_files: path to fiducial aligned folder
    num_hybcycles: number of total hyb cycles
    num_channels: number of total channels
    savefig: bool to write plot
    by_pos: bool to plot by pos or take average
    
    Returns
    -------
    matplotlib line plot
    """
    
    #path to write plots
    output_path = Path(path_to_files)
    
    #get error paths
    error_list = []
    for i in range(num_hybcycles):
        final_path = Path(path_to_files) / f"HybCycle_{i}"
        error_log = final_path.glob("*.txt")
        error_list.append(list(error_log))
    
    #read in error paths and convert to df
    error_log_df = []
    for i in range(len(error_list)):
        error_by_hyb = []
        for j in range(len(error_list[i])):
            error_df = pd.read_csv(error_list[i][j], sep = " ", header=None)
            pos = error_list[i][j].name.split("_")[1].replace("Pos","")
            error_df["pos"] = pos
            error_by_hyb.append(error_df)
        error_df_concat = pd.concat(error_by_hyb)
        error_df_concat.columns = ["channel","percent improvement", "nm off", "pos"]
        error_df_concat["nm off"] = error_df_concat["nm off"]*100
        error_df_concat["hyb"] = i
        error_log_df.append(error_df_concat)
        
    #combine final df
    error_final = pd.concat(error_log_df)
    if by_pos == False:
        #separate averaged distance off and percent improved by channel
        averaged_error_log_nm = []
        averaged_error_log_improved = []
        for c in range(num_channels):
            channel_off = []
            channel_improved = []
            for i in range(num_hybcycles):
                log = error_final[error_final["hyb"] == i].groupby("channel").mean()
                off = log["nm off"][c]
                imp = log["percent improvement"][c]
                channel_off.append(off)
                channel_improved.append(imp)
            averaged_error_log_nm.append(channel_off)
            averaged_error_log_improved.append(channel_improved)

        #plot error information
        color = ["red", "orange", "green", "blue"]
        channel = np.arange(1,num_channels+1,1)
        for i in range(len(averaged_error_log_nm)):
            plt.plot(np.arange(1,num_hybcycles+1,1), averaged_error_log_nm[i], color = color[i], label = f"Channel {channel[i]}")
            plt.xlabel("HybCycle", fontsize=12)
            plt.ylabel("nm off", fontsize=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            sns.despine()
        plt.legend()
        if savefig == True:
            plt.savefig(str(Path(output_path) / "distance_off.png"), dpi = 300)
        plt.show()


        for i in range(len(averaged_error_log_improved)):
            plt.plot(np.arange(1,num_hybcycles+1,1), averaged_error_log_improved[i], color = color[i], label = f"Channel {channel[i]}")
            plt.xlabel("HybCycle", fontsize=12)
            plt.ylabel("Percent Improved", fontsize=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            sns.despine()
        plt.legend()
        if savefig == True:
            plt.savefig(str(Path(output_path) / "percent_improved.png"), dpi = 300)
        plt.show()
    else:
        #separate averaged distance off and percent improved by channel
        error_log_list = []
        for c in range(num_channels):
            pos_list = []
            for p in error_final["pos"].unique():
                log = error_final[(error_final["channel"] == c) & (error_final["pos"] == p)]
                error_log_list.append(log)
        #plot error information
        color = ["red", "orange", "green", "blue"]
        for i in range(len(error_log_list)):
            sort_df = error_log_list[i].sort_values("hyb")
            channel_info = sort_df["channel"].iloc[0]
            plt.plot(sort_df["hyb"], sort_df["nm off"], color = color[channel_info],
                     label = f"Channel {channel_info+1}", lw=1)
            plt.xlabel("HybCycle", fontsize=12)
            plt.ylabel("nm off", fontsize=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            sns.despine()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if savefig == True:
            plt.savefig(str(Path(output_path) / "distance_off_all.png"), dpi = 300)
        plt.show()
        
        for i in range(len(error_log_list)):
            sort_df = error_log_list[i].sort_values("hyb")
            channel_info = sort_df["channel"].iloc[0]
            plt.plot(sort_df["hyb"], sort_df["percent improvement"], color = color[channel_info], 
                     label = f"Channel {channel_info+1}", lw=1)
            plt.xlabel("HybCycle", fontsize=12)
            plt.ylabel("percent improvement", fontsize=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            sns.despine()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if savefig == True:
            plt.savefig(str(Path(output_path) / "percent_improved_all.png"), dpi = 300)
        plt.show()
