"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 01/06/21
"""
#data management
from pathlib import Path
#image analysis
import tifffile as tf
from skimage.filters import threshold_otsu, threshold_local
#deep learning spot detection
from deepcell_spots.applications import Polaris
#general analysis
import time
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
#parallel processing
from concurrent.futures import ProcessPoolExecutor
#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

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
        
def find_spots(img_src, probability_threshold = 0.9, size_cutoff = 3):
    """
    This function will find dots using deepcell spots
    
    Parameters
    ----------
    img_src = path to image
    probability_threshold = parameter for deepcell spots
    size_cutoff = number of sigmas away from the mean for size to keep
    """
    
    #get hyb information
    hybcycle = Path(img_src).parent.name
    hyb_num = hybcycle.split("_")[1]
    
    #read image as z,c,x,y
    img = tf.imread(img_src)
    #reformat image if there is no z's
    if len(img.shape) == 3:
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    
    #use the pretrained model Polaris
    app = Polaris()
    #run for every channel except dapi
    channel_coords = []
    for c in range(img.shape[1]-1):
        image_c = img[:,c,:,:]
        coords = app.predict(np.reshape(image_c, (img.shape[0],img.shape[2],img.shape[3],1)),threshold=probability_threshold)
        channel_coords.append(coords)
       
    #convert into df
    df_list = []
    for channel in range(len(channel_coords)):
        for z in range(len(channel_coords[channel])):
            dots = pd.DataFrame(channel_coords[channel][z])
            dots.columns = ["y","x"]
            dots["z"] = z
            dots["ch"] = channel+1
            dots["hyb"] = hyb_num
            df_list.append(dots)
            
    #combine df
    df_final = pd.concat(df_list)
    
    #get dot characteristics
    area_list = []
    peak_int_list = []
    average_int_list = []
    for i in range(len(df_final)):
        x = int(df_final.iloc[i]["x"])
        y = int(df_final.iloc[i]["y"])
        z = int(df_final.iloc[i]["z"])
        c = int(df_final.iloc[i]["ch"])
        #get peak intensity of centroid
        peak_int_list.append(img[z,c-1,y,x])
        #get bounding box
        try:
            blob = get_region_around(img[z][c-1], center=[y,x], size=7, edge='raise')
        except IndexError:
            area_list.append(0)
            average_int_list.append(0)
            continue
        try:
            #estimate area of dot by local thresholding and summing boolean mask
            local_thresh = threshold_local(blob, block_size=7)
            label_local = (blob > local_thresh)
            area = np.sum(label_local)
            area_list.append(area)
            #also estimate average intensity of dot based on mask
            avg_int = np.sum((blob * label_local))/area
            average_int_list.append(avg_int)
        except:
            area_list.append(0)
            average_int_list.append(0)
        
    #add features
    df_final["size"] = area_list
    df_final["peak intensity"] = peak_int_list
    df_final["average intensity"] = average_int_list
    
    #filter by size
    mu, std = norm.fit(df_final["size"]) #fit gaussian to size dataset
    plt.hist(df_final["size"], density=True, bins=20)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x,p, label="Gaussian Fitted Data")
    plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
    plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
    plt.xlabel("Area by pixel")
    plt.ylabel("Proportion")
    plt.show()
    df_final = df_final[(df_final["size"] < (mu+(size_cutoff*std))) 
                                    & (df_final["size"] > (mu-(size_cutoff*std)))]
    
    #reorganize df
    df_final = df_final[["hyb","ch","x","y","z","size","peak intensity","average intensity"]]
    
    return df_final

def find_spots_parallel(img_list, probability_threshold = 0.9, 
                        size_cutoff = 3, output_folder="", encoded_within_channel=False):
    """
    This function will run deep cell spots in parallel for each hyb.
    
    img_list = list of paths for images
    probability_threshold = parameter for deepcell spots
    size_cutoff = number of sigmas away from the mean for size to keep
    output_folder = path to where you want the output
    encoded_within_channel = bool to split dots by channels
    """
    
    #start time
    start = time.time()
    #run parallel processing per hyb
    with ProcessPoolExecutor(max_workers=100) as exe:
        futures = []
        for img_src in img_list:
            fut = exe.submit(find_spots, img_src, probability_threshold, size_cutoff)
            futures.append(fut)

    #collect result from futures objects
    result_list = [fut.result() for fut in futures]
    
    #concatenate all hybs
    combined_df = pd.concat(result_list)
    combined_df = combined_df.reset_index(drop=True)

    #write csv
    if encoded_within_channel == True:
        #make output directory
        pos = Path(img_list[0]).name.split("_")[1]
        pos = pos.split(".")[0]
        new_output_folder = Path(output_folder) / pos
        for c in combined_df["ch"].unique():
            new_output_folder = new_output_folder / f"Channel_{c}"
            new_output_folder.mkdir(parents=True, exist_ok=True)
            for z in combined_df["z"].unique():
                combined_df_cz = combined_df[(combined_df["ch"]==c) & (combined_df["z"]==z)].reset_index(drop=True)
                combined_df_cz.to_csv(str(new_output_folder / f"locations_z_{z}.csv"))
    else:
        #make output directory
        pos = Path(img_list[0]).name.split("_")[1]
        pos = pos.split(".")[0]
        new_output_folder = Path(output_folder) / pos
        new_output_folder.mkdir(parents=True, exist_ok=True)
        for z in combined_df["z"].unique():
            combined_df_z = combined_df[combined_df["z"]==z].reset_index(drop=True)
            combined_df_z.to_csv(str(Path(new_output_folder)/ f"locations_z_{z}.csv"))
    
    print(f"This task took {(time.time()-start)/60} min")
    
    
    
    