"""
author: Katsuya Lex Colon
updated: 05/19/22
"""
#data management
import os
from pathlib import Path
from util import pil_imread
#image analysis
from skimage.filters import threshold_local
from photutils.detection import DAOStarFinder
import cv2
#general analysis
import numpy as np
import pandas as pd
from scipy.stats import norm
#plotting packages
import matplotlib.pyplot as plt
#parallel processing
from concurrent.futures import ProcessPoolExecutor
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
    
    return sources

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
        try:
            dots = len(daofinder(data,  threshold, fwhm))
        except TypeError:
            continue
        counts.append(dots)
    #find index with largest counts
    if len(counts) == 0:
        #just return a number
        return 4
    else:
        best_index = np.argmax(counts)
        #this is the best fwhm
        best_fwhm = fwhm_range[best_index]

        return best_fwhm
           
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
    
def dot_detection(img_src, HybCycle=0, size_cutoff=3, 
                  threshold=0.02,channel=1, num_channels=4):
    
    """
    Perform dot detection on image using daostarfinder.
    
    Parameters
    ----------
    img_src = path to image
    HybCycle = which hybcycle the image belongs to
    size_cutoff = number of standard deviation away from mean size area
    threshold = absolute pixel intensity the spot must be greater than
    num_channels = number of channels expected
    swapaxes = bool to flip channel and z axis
    
    Returns
    ----------
    locations csv file and size distribution plots
    """      
    
    #read image
    img = pil_imread(img_src, swapaxes=True)
    if img.shape[1] != num_channels:
        img = pil_imread(img_src, swapaxes=False)
        if img.shape[0] == img.shape[1]:
            img = check_axis(img)

    #using daostarfinder detection
    if len(img.shape)==3:
        #get optimal fwhm
        fwhm = get_optimum_fwhm(img[channel-1], threshold=threshold)
        #dot detect
        peaks = daofinder(img[channel-1], threshold=threshold,fwhm=fwhm)
        #if None was returned then return empty df
        try:
            peaks = peaks.to_pandas()
        except AttributeError:
            return pd.DataFrame()
        peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
        ch = np.zeros(len(peaks))+channel
        z_slice = np.zeros(len(peaks))
        peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
        peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
        dots = peaks
    else:
        dots = []
        for z in range(img.shape[0]):
            #get optimal fwhm
            fwhm = get_optimum_fwhm(img[z][channel-1], threshold=threshold)
            #dot detect
            peaks = daofinder(img[z][channel-1], threshold=threshold, fwhm=fwhm)
            #if None was returned for a particular z then continue
            try:
                peaks = peaks.to_pandas()
            except AttributeError:
                continue
            peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
            ch = np.zeros(len(peaks))+channel
            z_slice = np.zeros(len(peaks))+z
            peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
            peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
            dots.append(peaks)
        #check if combined df is empty
        if len(dots) == 0:
            return pd.DataFrame()
        else:
            dots = np.concatenate(dots)

    #make df and reorganize        
    dots = pd.DataFrame(dots)
    dots.columns = ["x", "y", "flux", "max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "ch", "z"]
    dots["hyb"] = HybCycle
    dots = dots[["hyb","ch","x","y","z", "flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits"]]

    #get area
    #subtract 1 from channels to get right slice
    coord = dots[["x","y","z","ch"]].values
    area_list = []
    for i in coord:
        x = int(i[0])
        y = int(i[1])
        z = int(i[2])
        c = int(i[3])
        #get bounding box
        try:
            if len(img.shape)==3:
                blob = get_region_around(img[c-1], center=[y,x], size=7, edge='raise')
            else:
                blob = get_region_around(img[z][c-1], center=[y,x], size=7, edge='raise')
        except IndexError:
            area_list.append(0)
            continue
        #estimate area of dot by local thresholding and summing binary mask
        try:
            local_thresh = threshold_local(blob, block_size=7)
            label_local = (blob > local_thresh)
            area = np.sum(label_local)
            area_list.append(area)
        except ValueError:
            area_list.append(0)
        except IndexError:
            area_list.append(0)

    #construct final df
    dots["size"] = area_list
    dots = dots[["hyb","ch","x","y","z","flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "size"]]

    if size_cutoff != None:
        #filter by size
        mu, std = norm.fit(dots["size"]) #fit gaussian to size dataset
        dots = dots[(dots["size"] < (mu+(size_cutoff*std))) 
                                & (dots["size"] > (mu-(size_cutoff*std)))]
        plt.hist(dots["size"], density=True, bins=20)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x,p, label="Gaussian Fitted Data")
        plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
        plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
        plt.xlabel("Area by pixel")
        plt.ylabel("Density")
    else:
        #filter by size
        mu, std = norm.fit(dots["size"]) #fit gaussian to size dataset
        plt.hist(dots["size"], density=True, bins=20)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x,p, label="Gaussian Fitted Data")
        plt.xlabel("Area by pixel")
        plt.ylabel("Density")

    plt.show()
    plt.clf()

    return dots
 
def dot_detection_parallel(img_src, size_cutoff=3, threshold=0.02,
                           channel=1,  num_channels=4):
    """
    This function will run dot detection in parallel, provided a list of images.
    
    Parameters
    ----------
    img_src= path to images where the pos is the same but has all hybcycles
    size_cutoff = number of standard deviation away from mean size area
    threshold = absolute pixel intensity the spot must be greater than
    channel = which channel to look at (1-4)
    num_channels = number of channels expected
    
    Returns
    -------
    locations csv file and size distribution plots
    """
    import time
    start = time.time() 
    
    #set output paths
    parent = Path(img_src[0]).parent
    while "notebook_pyfiles" not in os.listdir(parent):
        parent = parent.parent
    output_folder = parent / "notebook_pyfiles"/ "dots_detected"/ f"Channel_{channel}" 
    output_folder.mkdir(parents=True, exist_ok=True)
    
    #how many files
    print(f"Reading {len(img_src)} files and performing dot detection...")
    
    with ProcessPoolExecutor(max_workers=32) as exe:
        futures = []
        for img in img_src:
            #get hybcycle number
            img_parent_cycle = Path(img).parent.name
            HybCycle_mod = int(img_parent_cycle.split("_")[1])
            #dot detect
            fut = exe.submit(dot_detection, img_src=img, HybCycle=HybCycle_mod, size_cutoff=size_cutoff,
                             threshold=threshold,channel=channel, num_channels=num_channels)
            futures.append(fut)
            
    #collect result from futures objects
    result_list = [fut.result() for fut in futures]
    
    #concat df
    combined_df = pd.concat(result_list)
    
    #get number of z's
    num_z = combined_df["z"].unique()
    
    #get pos info
    pos = Path(img_src[0]).name.split("_")[1].replace(".ome.tif","")
    
    #output files
    for z in num_z:
        combined_df_z = combined_df[combined_df["z"]==z]
        output_path = output_folder /pos
        output_path.mkdir(parents=True, exist_ok=True)
        combined_df_z.to_csv(str(output_path) +f"/locations_z_{int(z)}.csv", index=False)
        
    print(f"This task took {round((time.time()-start)/60,2)} min")
