"""
Author: Katsuya Lex Colon
Date: 11/27/23
"""
#data management
import os
from pathlib import Path
from util import pil_imread
#general packages
from skimage.feature import peak_local_max
from radial_center import *
import pandas as pd
import cv2
#parallel processing
from concurrent.futures import ProcessPoolExecutor

def check_axis(img):
    """
    Determine if the img axis needs to be flipped if both channel and z axis is the same
    
    Parameters
    ----------
    img: 
        numpy 4d array
    
    Returns
    -------
    img:
        numpy 4d array
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

def get_region_around(im, center, size, edge='return'):
    """
    This function will generate a 2D cutout around detected spots.
    
    Parameters
    ----------
    im: 
        image tiff
    center: 
        x,y centers from dot detection
    size: 
        size of bounding box
    edge:
        "raise" will output error message if dot is at border and
        "return" will adjust bounding box 
            
    Returns
    -------
    region: 
        2D numpy array 
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

def centroid_offset(x, y, x_g, y_g, region_size):
    """
    Returns corrected centroids.

    Parameters
    ----------
    x:
        uncorrected x coord
    y:
        uncorrected y coord
    x_g:
        fitted x 
    y_g:
        fitted y
    region_size:
        size of bounding box
    
    Returns
    -------
    x,y : 
        coordinates as floats
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


def calculate_sharpness_laplacian(image):
    """
    Calculate sharpness of spot looking at variance of laplacian.
    Parameters
    ----------
    image:
        2d numpy array of isolated spot
    Returns
    -------
    sharpness:
        np.float
    """
    
    #calculate laplacian of the 2d cutout
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    #obtain the variance
    #sharp images will have large variance
    sharpness = np.var(laplacian)
    
    return sharpness

def calculate_sharpness_tenengrad(image, ksize=3):
    """
    Calculate sharpness of spot looking at magnitude of sobel gradients.
    Parameters
    ----------
    image:
        2d numpy array of isolated spot
    Returns
    -------
    sharpness:
        np.float
    """
    #calculate sobel gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    #take sum of squared gradients as a measure of sharpness
    #larger gradients should have greater sharpness
    sharpness = np.sum(gx ** 2 + gy ** 2)
    
    return sharpness

def calculate_bilateral_symmetry(image):
    """
    Calculate the bilateral mirror symmetry of an image.

    Parameters
    -----------
    image (numpy.ndarray): 
        A 2D numpy array representing the grayscale image.

    Returns
    -----------
    float: 
        A symmetry score, with lower values indicating higher symmetry.
    """
    # Split the image into left and right halves
    mid = image.shape[1] // 2
    left_half = image[:, :mid]
    right_half = image[:, mid:]

    # If the image has an odd number of columns, crop one column from the right half
    if left_half.shape[1] != right_half.shape[1]:
        right_half = right_half[:, :-1]

    # Flip the right half horizontally
    right_half_flipped = np.flip(right_half, axis=1)

    # Calculate the absolute difference between the left half and the flipped right half
    difference = np.abs(left_half - right_half_flipped)

    # Compute a symmetry score (mean of differences)
    symmetry_score = np.mean(difference)

    return symmetry_score

def detect_spot(image, HybCycle = 0, threshold = 0.05, region_size = 7):
    """
    A function to detect spots, perform radial centering for subpixel centroids, and obtain spot features.
    Parameters
    ----------
    image:
        4d numpy array of your image (z,c,y,x)
    threshold:
        threshold value during peak local max spot detection
    region_size:
        size of the bounding box.
    
    Returns
    --------
    df: locations of spots along with features
    """

    #find spots then radial center
    centroids = []
    for z in range(image.shape[0]):
        for ch in range(image.shape[1]-1):
            dot_cands = peak_local_max(image[z, ch], threshold_abs = threshold, min_distance = 1)
            for coord in dot_cands:
                #get 2d cutout
                im_data = get_region_around(image[z, ch], coord, region_size)
                
                #calculate sharpness by laplacian
                sh_l = calculate_sharpness_laplacian(im_data)
                
                #calculate sharpness by tenegrad
                sh_t = calculate_sharpness_tenengrad(im_data, ksize=3)
                
                #calculate symmetry
                sym_score = calculate_bilateral_symmetry(im_data)

                #obtain center
                x_r, y_r, sigma = radialcenter(im_data)

                #get corrected coordinates
                x, y = centroid_offset(coord[1], coord[0], x_r, y_r, region_size)

                #calculate max intensity
                peak_intensity = image[z, ch][coord[0], coord[1]]

                centroids.append([x, y, z, ch+1, sigma, peak_intensity, sh_l, sh_t, sym_score])
    
    #convert to pandas df
    df = pd.DataFrame(centroids)
    #make sure to normalize sharpness
    #include two symmetry measures, and two sharpness measures
    df.columns = ["x", "y", "z", "ch", "sigma", "peak intensity", "sharpness_l", "sharpness_t", "symmetry"]
    df.insert(0, "hyb", HybCycle)
    
    return df

def dot_detection_radial_center(img_src, threshold=0.02, num_channels=4):
    """
    This function will run dot detection in parallel, provided a list of images.
    
    Parameters
    ----------
    img_src (str):
        path to images where the pos is the same but has all hybcycles
    threshold(float or int):
        threshold for pixel intensity
    num_channels (int):
        number of channels in image
    
    Returns
    -------
    locations csv file and size distribution plots
    """
    import time
    start = time.time() 
    
    #set output paths
    parent = Path(img_src[0]).parent
    while "pyfish_tools" not in os.listdir(parent):
        parent = parent.parent
    output_folder = parent / "pyfish_tools"/ "output" / "dots_detected_radial_centered" 
    output_folder.mkdir(parents=True, exist_ok=True)
    
    #how many files
    print(f"Reading {len(img_src)} files and performing dot detection...")
    
    with ProcessPoolExecutor(max_workers=32) as exe:
        futures = []
        for img in img_src:
            
            #get hybcycle number
            img_parent_cycle = Path(img).parent.name
            HybCycle = int(img_parent_cycle.split("_")[1])
            
            #read image
            image = pil_imread(img, swapaxes=True)
            if image.shape[0] == image.shape[1]:
                image = check_axis(img)
            if image.shape[1] != num_channels:
                image = pil_imread(img, swapaxes=False)
                
            #dot detect
            fut = exe.submit(detect_spot, image=image, HybCycle=HybCycle,threshold=threshold,
                             region_size=7)
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
    for ch in range(num_channels-1):
        for z in num_z:
            combined_df_z_ch = combined_df[(combined_df["z"]==z) & (combined_df["ch"]==ch+1)].reset_index(drop=True)
            output_path = output_folder / f"Channel_{ch+1}" / pos
            output_path.mkdir(parents=True, exist_ok=True)
            combined_df_z_ch.to_csv(str(output_path) +f"/locations_z_{int(z)}.csv", index=False)
        
    print(f"This task took {round((time.time()-start)/60,2)} min")