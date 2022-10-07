"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 06/02/22
"""
#data management
import os
from pathlib import Path
from util import pil_imread
#image analysis
from skimage.filters import threshold_local
from photutils.detection import DAOStarFinder
#general analysis
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from collections import Counter
#parallel processing
from concurrent.futures import ProcessPoolExecutor
#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def colocalizing_dots(df, radius=2, ch_seed=1):
    """
    Performs nearest neighbor search provided a given search radius.
    If the nearest neighbor has a euclidean pixel distance <= radius then the dots are colocalizing.
    Parameters
    ----------
    df: first set of dots
    radius: search radius
    ch_seed: which channel to use as seed first
    """
    
    #reset index for df just in case
    df = df.reset_index(drop=True)
    
    #check num channels
    num_channels = len(df["ch"].unique())
    
    #using sklearn nearest neighbor algorithm to find nearest dots
    #initialize algorithm
    neigh = NearestNeighbors(n_neighbors=num_channels, radius=radius, metric="euclidean", n_jobs=1)
    
    #rename seed
    initial_seed = df[df["ch"] == ch_seed][["x","y"]]
    
    #check if df is empty
    if len(initial_seed) == 0:
        return []
    
    #store index of channel seed
    initial_seed_index = initial_seed.index.tolist()

    #remove channel that is being compared against in list
    df = df[df["ch"] != ch_seed]

    #find neighbors at a given search radius across channnel
    neighbor_list = []
    for i in df["ch"].unique():
        #initialize neighbor
        neigh.fit(df[df["ch"] == i][["x","y"]])
        #find neighbors
        _,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
        neighbor_list.append(neighbors)
    
   #organize dots so that each channel is pooled together
    neighbor_list2 = []
    for i in range(len(neighbor_list[0])):
        temp = []
        #loop through channels for same position
        for j in range(len(neighbor_list)):
            temp.append(neighbor_list[j][i].tolist())
        #convert to original index on df
        orig_idx = []
        for ch in range(len(df["ch"].unique())):
            ch_pool = []
            for idx in temp[ch]:
                try:
                    ch_pool.append(df[df["ch"] == df["ch"].unique()[ch]].iloc[idx].name)
                except IndexError:
                    ch_pool.append([])
            orig_idx.append(ch_pool)
        #add index of dot being analyzed
        orig_idx.insert(0,[initial_seed_index[i]])
        #remove empty lists which correspond to missed dots in barcoding round
        orig_idx = [sublist for sublist in orig_idx if sublist != []]
        neighbor_list2.append(orig_idx)

    #flatten list and take closest spot
    neigh_flat = []
    for sublist in neighbor_list2:
        closest = []
        for element in sublist:
            closest.append(element[0])
        neigh_flat.append(closest)
    
    return neigh_flat

def daofinder(data,  threshold, fwhm = 4.0):
    """
    This function will return the output of daostarfinder
    Parameters
    ----------
    data:  2D array
    threshold: absolute intensity threshold
    fwhm: full width half maximum
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
    data: 2D array
    threshold:  initial threshold for testing
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
    im: image tiff
    center:  x,y centers from dot detection
    size: size of bounding box
    edge: "raise" will output error message if dot is at border and
            "return" will adjust bounding box 
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
    
def dot_detection(img, HybCycle=0, size_cutoff=None,threshold=500):
    
    """
    Perform dot detection on image using daostarfinder.
    
    Parameters
    ----------
    img: image array
    HybCycle: which hybcycle the image belongs to
    size_cutoff: number of standard deviation away from mean size area
    threshold: absolute pixel intensity the spot must be greater than
    """      

    #using daostarfinder detection
    if len(img.shape)==3:
        #reshape image if z axis is missing
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
    print(f"Image shape: {img.shape}")
    dots_list = []
    for channel in range(img.shape[1]):
        z_stack = []
        for z in range(img.shape[0]):
            #get optimal fwhm
            fwhm = get_optimum_fwhm(img[z][channel], threshold=threshold)
            #dot detect
            peaks = daofinder(img[z][channel], threshold=threshold, fwhm=fwhm)
            #if None was returned for a particular z then continue
            try:
                peaks = peaks.to_pandas()
            except AttributeError:
                continue
            peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]]
            ch = np.zeros(len(peaks))+(channel+1)
            z_slice = np.zeros(len(peaks))+(z)
            peaks["ch"] = ch
            peaks["z"] = z_slice
            z_stack.append(peaks)
        #combine z for same channel
        try:
            z_df = pd.concat(z_stack).reset_index(drop=True)
        except:
            continue
        dots_list.append(z_df)
            
    #check if df_list is empty, if so then no dots were detected at certain threshold
    if len(dots_list) == 0:
        return pd.DataFrame()
    
    #make df and reorganize   
    final_dots = []
    for dot_ch in dots_list:
        for z in dot_ch["z"].unique():
            dots = dot_ch[dot_ch["z"]==z]
            dots.columns = ["x", "y", "flux", "max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "ch", "z"]
            dots["hyb"] = HybCycle
            dots = dots[["hyb","ch","x","y","z", "flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits"]]

            #get area and intensity across channels
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
            #filter spots by size if desired
            if size_cutoff != None:
                #filter by size
                mu, std = norm.fit(dots["size"]) #fit gaussian to size dataset
                dots = dots[(dots["size"] < (mu+(size_cutoff*std))) 
                                        & (dots["size"] > (mu-(size_cutoff*std)))]
                
            final_dots.append(dots)
            
    #generate final for colocalization
    final_dots = pd.concat(final_dots).reset_index(drop=True)
    
    return final_dots

def ratiometric_dot_detection(img_src, HybCycle=0, size_cutoff=None,
                  threshold=500, radius = 2, num_channels=4):
    """
    This function will find colocalizing spots across channels for ratiometric analysis.
    
    Parameters
    ----------
    img_src: dot locations file
    HybCycle: which hybcycle
    size_cutoff: number of std away from dot size mean
    threshold: int value that the amplitude must be above
    radius: radius search in pixels
    num_channels: number of channels in image
    """
    
    #read image
    img = pil_imread(img_src, swapaxes=True)
    if img.shape[1] != num_channels:
        img = pil_imread(img_src, swapaxes=False)
        
    #exclude dapi channel
    img = img[:,:img.shape[1]-1,:,:]
        
    #detect dots for each channel
    dots = dot_detection(img, HybCycle=HybCycle, size_cutoff=size_cutoff, threshold=threshold)
    
    #get number of channels
    num_channels = img.shape[1]
    
    final_coloc_spots = []
    
    for z in dots["z"].unique():
        dots_z = dots[dots["z"] == z].reset_index(drop=True)
        #parallel processing for nearest neighbor computation for each channel
        with ProcessPoolExecutor(max_workers=4) as exe:
            futures = []
            for i in np.arange(1,num_channels+1,1):
                fut = exe.submit(colocalizing_dots, dots_z,
                                 radius=radius, ch_seed=i)
                futures.append(fut)

        #collect result from futures objects
        result_list = [fut.result() for fut in futures]
        
        #remove empty results
        result_list = [sublist for sublist in result_list if sublist != []]

        #flatten and convert to frozensets
        all_index = [frozenset(element) for sublist in result_list for element in sublist]

        #get unique dot sets 
        obj = Counter(all_index)

        #convert to df
        df = pd.DataFrame.from_dict(obj, orient='index').reset_index()

        #get indicies
        spots = df["index"].values

        #generate final ratiometric spots
        final_rat_spots = []
        #keep track of spots
        spot_history = set()
        for i in range(len(spots)):
            #if there is a spot index used already then skip
            if spots[i] & spot_history:
                continue
            #isolate spot info and sort by channel number
            rat_spot = dots_z.iloc[list(spots[i])].sort_values("ch")
            #get average xy coord
            mean_xy = rat_spot[["x","y"]].mean().to_list()
            #get z info
            z = [rat_spot["z"].iloc[0]]
            #get average features for each spot
            features = rat_spot[["flux","sharpness", "roundness by gaussian fits", "size"]].mean().to_list()
            #get intensity for each channel
            int_ch = rat_spot["max intensity"].astype(int).to_list()
            #spots colocalized
            num_spots = [len(rat_spot["ch"])]
            #check which channels are present
            missing_ch = list(set(np.arange(1,num_channels+1,1)) - set(rat_spot["ch"]))
            #for missing channels go to average xy and retrieve intensity
            for ch in missing_ch:
                if len(img.shape) == 3:
                    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
                img_ch = img[int(z[0])][ch-1]
                blank_int = img_ch[int(mean_xy[1]),int(mean_xy[0])]
                int_ch.insert(ch-1,blank_int)
            #combine all info
            rat_spot_final = mean_xy+z+features+num_spots+int_ch
            final_rat_spots.append(rat_spot_final)
            #add spots to history
            for _id in spots[i]:
                spot_history.add(_id)

        #create df    
        final_df = pd.DataFrame(final_rat_spots)

        #create names for columns
        column_names = (["x","y","z","flux","sharpness","roundness by gaussian fits", "size","spots colocalized"] 
                        + [f"ch{i} intensity" for i in np.arange(1,num_channels+1,1)])

        #add column names
        final_df.columns = column_names
        final_df.insert(0,"hyb",HybCycle)
        
        final_coloc_spots.append(final_df)
        
    #colocalizing spots
    coloc_spots = pd.concat(final_coloc_spots).reset_index(drop=True)
    
    return coloc_spots

def pixel_based_detection(img_src, HybCycle=0,threshold=500, 
                          size_cutoff=None, num_channels=4):
    
    """
    This function will collapse all channels and dot detect to get spot coordinates. 
    Pixel intensity for each spot will then be obtain across channels. 
    
    Parameters
    ----------
    img_src: dot locations file
    HybCycle: which hybcycle
    threshold: int value that the amplitude must be above
    size_cutoff: number of std away from dot size mean
    num_channels: number of channels in image
    """
    
    #read image
    img = pil_imread(img_src, swapaxes=True)
    if img.shape[1] != num_channels:
        img = pil_imread(img_src, swapaxes=False)
        
    #exclude dapi channel
    img = img[:,:img.shape[1]-1,:,:]
     
    #max project channels
    img_max = np.max(img, axis=1)
    
    #get all coord
    dots = dot_detection(img_max, HybCycle=HybCycle, threshold=threshold, size_cutoff=size_cutoff)
    xyz = dots[["x","y","z"]]

    #get int for each channel by pixel
    int_per_spot = []
    for i in range(len(xyz)):
        iso_xyz = xyz.iloc[i].tolist()
        x = int(iso_xyz[0])
        y = int(iso_xyz[1])
        z = int(iso_xyz[2])
        ch_int = []
        for ch in range(img.shape[1]):
            _int = img[z,ch,y,x]
            ch_int.append(_int)
        all_info = iso_xyz + ch_int
        int_per_spot.append(all_info)
        
    #convert to df
    spot_info = pd.DataFrame(int_per_spot)
    #channel names
    channel_names = [f"ch{i+1} intensity" for i in range(img.shape[1])]
    column_names = ["x","y","z"] + channel_names 
    spot_info.columns = column_names
    spot_info.insert(0,"hyb",HybCycle)
    
    return spot_info
        
def ratiometric_dot_detection_parallel(img_src, size_cutoff=None, threshold=500,
                                       radius = 1.5, pixel_based = False, num_channels=4):
        
    """
    This function will run ratiometric dot detection in parallel, provided a list of images.
    
    Parameters
    ----------
    img_src: path to images where the pos is the same but has all hybcycles
    size_cutoff: number of standard deviation away from mean size area
    threshold: int value that the amplitude must be above
    radius: radius search in pixels
    pixel_based: do you want to perform pixel based intensity grabbing, otherwise
                 it will perform spot based colocalization
    num_channels: number of channels in image
    """
    
    import time
    start = time.time() 
    
    #set output paths
    parent = Path(img_src[0]).parent
    while "seqFISH_datapipeline" not in os.listdir(parent):
        parent = parent.parent
        
    if pixel_based == False:
        output_folder = parent / "seqFISH_datapipeline"/ "dots_detected" / "spot_based"
    else:
        output_folder = parent / "seqFISH_datapipeline"/ "dots_detected" / "pixel_based"
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
            if pixel_based == False:
                fut = exe.submit(ratiometric_dot_detection, img_src=img, HybCycle=HybCycle_mod, size_cutoff=size_cutoff,
                                 threshold=threshold,radius=radius,num_channels=num_channels)
                futures.append(fut)
            else:
                fut = exe.submit(pixel_based_detection, img_src=img, HybCycle=HybCycle_mod, size_cutoff=size_cutoff,
                                 threshold=threshold,num_channels=num_channels)
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
