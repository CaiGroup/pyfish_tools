"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 01/06/22
"""
#data management
import os
from pathlib import Path
import glob
import re
from webfish_tools.util import find_matching_files
#image analysis
import tifffile as tf
from skimage.filters import threshold_otsu, threshold_local
from photutils.detection import DAOStarFinder
#general analysis
import numpy as np
import pandas as pd
from scipy.stats import norm
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#linear regression
from scipy.stats import linregress
#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def combine_dot_files(path_dots, hyb_start=0,hyb_end=63,num_HybCycle=32, pos= 0, 
                      channel = 1, num_z= 1, opt_files = False):
    """Function to read in all dot files and make combined locations csv
    Parameters
    ----------
    path_dots = path to dots_detected folder
    hyb_start = which hyb cycle is the begining
    hyb_end = which hybcycle is the end
    num_HybCYcle= number of total hyb cycles
    pos = position number
    channel = specific channel (1-4)
    num_z = number of zs
    opt_files = get optimization files
    
    Returns
    -------
    writes combined dot file into dots_comb folder
    """
    if opt_files == True:
        dot_path_list = []
        #get dot location file paths 
        for i in np.arange(hyb_start, hyb_end+1, 1):
            dots_folders= Path(path_dots) / f"HybCycle_{i}"
            dot_files = list(dots_folders.glob(f'MMStack_Pos{pos}_*.csv'))
            if dot_files == []:
                raise Exception(f"Missing Pos {pos} in HybCycle_{i}")
            #check if threshold starts with 0 and is a float
            is_float = str(dot_files[0]).split("_")[-1].replace(".csv","").split(".")
            #organize files numerically
            if len(is_float) == 2 and int(is_float[0]) == 0:
                sort_paths = sorted(dot_files)
            else:
                key = [int(re.search(f'Pos{pos}_(\\d+)*', str(f)).group(1)) for f in dot_files]
                sort_paths = list(np.array(dot_files)[np.argsort(key)])
            dot_path_list.append(sort_paths)

        #if num of thresholded csv is <11 then duplicate the lowest threshold
        for i in range(num_HybCycle):
            off = 11 - len(dot_path_list[i])
            if off == 0:
                continue
            else:
                for _ in range(off):
                    dot_path_list[i].insert(0,dot_path_list[i][0])

        #reorganize dot paths so that the threshold is matched for each hyb           
        reorg_dot_files = []
        for i in range(11):
            temp_list= []
            for j in range(num_HybCycle):
                temp_list.append(dot_path_list[j][i])
            reorg_dot_files.append(temp_list)
            
        #make textfile of new thresholds for later reference
        output_path_text = Path(reorg_dot_files[0][0]).parent.parent
        f = open(str(output_path_text) + f"/optimal_threshold_test_complete_ch{channel}.txt" ,"w+")
        for element in reorg_dot_files:
            for i in range(len(element)):
                csv_name = str(element[i].name)
                csv_name = csv_name.split("_")
                threshold_extracted = csv_name[2].replace(".csv","")
                f.write(threshold_extracted + " ")
            f.write("\n")
        f.close()
        
        #read files and concatenate
        df_list = []
        for i in range(11):
            df_read = [pd.read_csv(str(j), index_col=0) for j in reorg_dot_files[i]]
            df_con = pd.concat(df_read)
            df_list.append(df_con)
        #write files 
        for i in range(11):
            comb = df_list[i]
            comb_ch = comb[comb["ch"]==channel]
            for z in range(num_z):
                comb_z = comb_ch[comb_ch["z"]==z]
                output_folder = dot_path_list[0][0].parent.parent.parent.parent/ "dots_comb"
                output_path = output_folder  /f"Channel_{channel}"/f"MMStack_Pos{pos}"/f"Threshold_{i}"/"Dot_Locations"
                output_path.mkdir(parents=True, exist_ok=True)
                comb_z.to_csv(str(output_path) + f"/locations_z_{z}.csv", index=False)
               
    else:
        #specify position
        position_dots = f'MMStack_Pos{pos}.csv'
        #get all paths for specific position across hybs
        files_dots, _, _ = find_matching_files(path_dots, 'HybCycle_{hyb}' + f'/{position_dots}')
        files_dots = [str(f) for f in files_dots]

        dots = [pd.read_csv(j, index_col=0) for j in files_dots]
        con = pd.concat(dots)

        for z in range(num_z):
            con_z = con[con["z"]==z]
            output_folder = Path(files_dots[0]).parent.parent.parent.parent.parent / "dots_comb" /"final"
            output_path = output_folder /f"Channel_{channel}"/f"MMStack_Pos{pos}"
            output_path.mkdir(parents=True, exist_ok=True)
            con_z.to_csv(str(output_path) +f"/locations_z_{z}.csv", index=False)
            
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
        dots = len(daofinder(data,  threshold, fwhm))
        counts.append(dots)
    #find index with largest counts
    best_index = np.argmax(counts)
    #this is the best fwhm
    best_fwhm = fwhm_range[best_index]
    
    return best_fwhm

def dot_counts(img_src, threshold = 200, fwhm=4.0, channel = 1):
    """
    A function to count the number of dots found at a certain threshold
    Parameters
    ----------
    img_src= path to image
    threshold = absolute threshold
    fwhm = full width half maximum
    channel = which channel to analyze
    
    Returns
    -------
    textfile containing counts detected at a certain threshold
    """
    
    #make output folder
    img_parent = Path(img_src).parent.parent.parent
    hyb_cycle = Path(img_src).parent.name
    pos_name = str(Path(img_src).name).replace(".ome.tif","") 
    output_path = Path(img_parent) / "threshold_counts" / f"Channel_{channel}" / hyb_cycle / pos_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    img = tf.imread(img_src)
    
    #dot detection across z
    counts = 0
    if len(img.shape)== 3:
        peaks = daofinder(img[channel-1], threshold=threshold, fwhm=fwhm)
        counts += len(peaks)
        _list = [threshold, counts]
    else: 
        for z in range(img.shape[0]):
            peaks = daofinder(img[z][channel-1], threshold=threshold,fwhm=fwhm)
            counts += len(peaks)
        _list = [threshold, counts]
    
    #make textfile
    f = open(str(output_path) + f"/thresh{threshold}_counts.txt" ,"w+")
    
    #write textfile
    for element in _list:
        f.write(str(element) + "\n")
    f.close()
    
def sliding_lineregress(arr, window=10, reduce_cutoff = 2):
    """
    A sliding linear regression function to obtain various slopes
    Parameters
    ----------
    arr = array of y values
    window = window size
    reduce_cutoff = how many windows to go back
    
    Returns
    -------
    the best threshold index
    """
    #collect slopes
    slope_list = []
    #define window
    win = np.arange(0,len(arr), window).astype(int)
    #go through data points and fit line
    for i in range(len(win)-1):
        y = arr[win[i]:win[i+1]]
        x = np.arange(1, len(y)+1,1)
        slope, _, _, _, _ = linregress(x,y)
        slope_list.append(np.abs(slope))
    #identify index with minimum slope 
    min_idx = (np.argmin(slope_list)-reduce_cutoff)*window
    
    return min_idx
    
def find_threshold(img_src, threshold_min = 100, 
                   threshold_max= 1000, interval=100, 
                   HybCycle = 0, channel = 1, pos_list=None, 
                   reduce_cutoff= 2, window=10):
    
    """This function will find the optimal threshold to use during peak local max detection
    Parameters
    ----------
    img_src= path to image
    threshold_min = starting threshold
    threshold_max = ending threshold
    interval = spacing between maximum and minimum
    HybCycle= which hybcycle the image belongs to
    channel = which channel to look at (1-4)
    pos_list = list of position numbers
    reduce_cutoff = number of windows to go back in sliding_lineregress
    window = window size for sliding_lineregress
    
    Returns
    ----------
    optimum threshold text file
    percent difference plot
    number of dots detected plot
    combined csv containing number of dots found at each threshold
    """
    
    #generate list of thresholds
    thresh_list = np.linspace(threshold_min,threshold_max, num = interval)
    thresh_list = list(thresh_list.flatten())
    
    import time
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=20) as exe:
        futures = {}
        if type(img_src) == list:
            #get best fwhm
            fwhm_list=[]
            for path in img_src:
                test_data = tf.imread(path)
                if len(test_data.shape) == 3:
                    fwhm = get_optimum_fwhm(test_data[channel-1], threshold_min)
                    fwhm_list.append(fwhm)
                else:
                    #just check 1st z (image has to be z,c,x,y)
                    fwhm = get_optimum_fwhm(test_data[0][channel-1], threshold_min)
                    fwhm_list.append(fwhm)
            #use median fwhm
            fwhm = np.median(fwhm_list)
            
            for image in img_src:
                for thresh in thresh_list:
                    fut = exe.submit(dot_counts, image, thresh, fwhm, channel)
                futures[fut] = image
                
            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {time.time() - start} seconds')
                
            print("completed tasks = ", len(futures))
        else:
            #get best fwhm
            test_data = tf.imread(img_src)
            if len(test_data.shape) == 3:
                fwhm = get_optimum_fwhm(test_data[channel-1], threshold_min)
            else:
                #just check 1st z (image has to be z,c,x,y)
                fwhm = get_optimum_fwhm(test_data[0][channel-1], threshold_min)
            for thresh in thresh_list:
                fut = exe.submit(dot_counts, img_src, thresh, fwhm, channel)
                futures[fut] = thresh

            for fut in as_completed(futures):
                thresh = futures[fut]
                print(f'Threshold {thresh} completed after {time.time() - start} seconds')
            print("completed tasks = ", len(futures))
            
    for _ in pos_list:
        #read in textfiles
        if type(img_src)!=list:
            img_parent = Path(img_src).parent.parent.parent
        else:
            img_parent = Path(img_src[0]).parent.parent.parent
        output_path = Path(img_parent) / "threshold_counts" /f"Channel_{channel}" / f"HybCycle_{HybCycle}" /f"MMStack_Pos{_}"
        paths = glob.glob(str(output_path) +"/*_counts.txt")

        data_list= []
        for i in paths:
            data = pd.read_csv(i, sep=" ", header=None).T
            data_list.append(data)

        #combine text files to get final csv
        if len(data_list)==1:
            final=data_list[0]
        else:
            final = pd.concat(data_list)
        final.columns = ["Threshold", "Total Counts"]
        final = final.sort_values("Threshold")
        final = final.reset_index(drop=True)
        final.to_csv(str(output_path) + f"/combined_thresh_HybCycle{HybCycle}.csv")

        #remove text files
        files_all = os.listdir(str(output_path))
        for item in files_all:
            if item.endswith(".txt"):
                os.remove(os.path.join(str(output_path), item))

        #calculate percent change
        vector_sum = final["Total Counts"].values[:-1] + final["Total Counts"].values[1:]
        percent_change = np.abs(np.diff(final["Total Counts"]))/(vector_sum/2)

        # sliding linear regression to find region where slope is minimum 
        idx = sliding_lineregress(percent_change, window=window, reduce_cutoff=reduce_cutoff)

        #get threshold at inflection
        thresh = final["Threshold"].iloc[idx]

        #get counts at threshold
        counts_at_thresh = final["Total Counts"].iloc[idx]

        #generate 10 more equally spaced thresholds before initial
        #first lets get max counts
        max_counts = np.max(final["Total Counts"])
        
        #generate range of counts starting from initial threshold to max counts
        counts_range = np.linspace(counts_at_thresh, max_counts, 11)
        
        #store thresholds
        thresh_counts = []
        #keep intitial values
        thresh_counts.append([thresh, counts_at_thresh, "initial"])
        for i in np.arange(1,11,1):
            if i < 10:
                new_idx = np.argwhere(final["Total Counts"].values >= counts_range[i])
                new_idx = new_idx[len(new_idx)-1][0]
                thresh_new = final["Threshold"].iloc[new_idx]
                counts_at_thresh_new = final["Total Counts"].iloc[new_idx]  
                thresh_counts.append([thresh_new, counts_at_thresh_new, counts_range[i]])
            else:
                thresh_new = final["Threshold"].iloc[0]
                counts_at_thresh_new = final["Total Counts"].iloc[0]  
                thresh_counts.append([thresh_new, counts_at_thresh_new, counts_range[i]])

        #plot percent difference        
        plt.plot(percent_change)
        plt.xlabel("Each Threshold", size=12)
        plt.ylabel("Percent Change", size=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        plt_idx = final[final["Threshold"] == thresh_counts[0][0]].index
        plt.axhline(percent_change[plt_idx[0]], linestyle= "--",linewidth = 1, c = "red")
        sns.despine()
        plt.savefig(str(output_path) + f"/percent_diff_HybCycle_{HybCycle}", dpi = 300)
        plt.show()
        plt.clf()

        #plot total counts
        plt.plot(final["Threshold"], final["Total Counts"])
        plt.xlabel("Threshold", size=12)
        plt.ylabel("Total Counts", size=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        for i in range(len(thresh_counts)):
            plt.axvline(thresh_counts[i][0], linestyle= "--",linewidth = 1, c = "red")
        sns.despine()
        plt.savefig(str(output_path) + f"/totalcounts_HybCycle_{HybCycle}", dpi = 300)
        plt.show()
        plt.clf()

        #make textfile
        f = open(str(output_path) + f"/optimal_threshold_HybCycle_{HybCycle}.txt" ,"w+")
        for element in thresh_counts:
            f.write(f"Threshold {element[0]} Counts {element[1]} Min_dots {element[2]} FWHM {fwhm}" + "\n")
        f.close()
        
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
    
def dot_detection(img_src, fwhm = 4.0, HybCycle=0, size_cutoff=3, 
                  opt_thresh=0.001,channel=1,pos=0,choose_thresh_set = 0,
                  optimize=False, output=False):
    
    """
    This function uses DAOStarFinder with thresholds obtained from find_threshold(). 
    
    Parameters
    ----------
    img_src = path to image
    fwhm = full-width half maximum
    HybCycle = which hybcycle the image belongs to
    size_cutoff = number of standard deviation away from mean size area
    opt_thresh = threshold used during optimization
    channel = which channel to look at (1-4)
    pos = position number (used to get FWHM)
    choose_thresh_set = int for which threshold set you want to use
    optimize = bool to test different threshold and min dots
    output = bool to output files
    
    Returns
    ----------
    locations csv file and size distribution plots
    """      
    
    #set output paths
    img_parent = Path(img_src).parent.parent.parent
    output_folder = Path(img_parent) / "dots_detected"/ f"Channel_{channel}" / f"HybCycle_{HybCycle}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = str(output_folder / Path(img_src).name)
    output_path = output_path.replace(".ome.tif",".csv")
    
    #read image
    img = tf.imread(img_src)
    
    #get optimal threshold
    if optimize == True:
        #using daostarfinder detection
        if len(img.shape)==3:
            z=0
            peaks = daofinder(img[channel-1], threshold=opt_thresh,fwhm=fwhm)
            peaks = peaks.to_pandas()
            peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
            ch = np.zeros(len(peaks))+channel
            z_slice = np.zeros(len(peaks))+z
            peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
            peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
            dots = peaks
        else:
            dots = []
            for z in range(img.shape[0]):
                peaks = daofinder(img[z][channel-1], threshold=opt_thresh, fwhm=fwhm)
                peaks = peaks.to_pandas()
                peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
                ch = np.zeros(len(peaks))+channel
                z_slice = np.zeros(len(peaks))+z
                peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
                peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
                dots.append(peaks)
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
        dots = dots[["hyb","ch","x","y","z", "flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "size"]]
        #write out plot
        output_path_adj = output_path.replace(".csv", f"_{opt_thresh}.csv")
        if size_cutoff != None:
            #filter by size
            mu, std = norm.fit(dots["size"]) #fit gaussian to size dataset
            plt.hist(dots["size"], density=True, bins=20)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            dots = dots[(dots["size"] < (mu+(size_cutoff*std))) 
                                    & (dots["size"] > (mu-(size_cutoff*std)))]
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
        if output == True:
            plt.savefig(str(output_folder) + f"/size_hist_{opt_thresh}.png", dpi = 300)
        plt.show()
        plt.clf()
        #write out final filtered dots
        if output == True:
            dots.to_csv(output_path_adj)
        else:
            return dots
            
    else:
        #output folder
        img_parent = Path(img_src).parent.parent.parent
        output_folder = Path(img_parent) / "dots_detected" /"final" /f"Channel_{channel}"/f"HybCycle_{HybCycle}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = str(output_folder / Path(img_src).name)
        output_path = output_path.replace(".ome.tif",".csv")
        
        #get optimal threshold after decoding verification
        gen_path = img_parent / "dots_detected" /f"Channel_{channel}" /f"optimal_threshold_test_complete_ch{channel}.txt"
        opt_thresh = pd.read_csv(gen_path, sep="\t", header = None)
        #pick thresh set
        opt_thresh_set = opt_thresh[0][choose_thresh_set]

        #parse list of thresholds and pick correct one for each hyb
        opt_thresh_set = opt_thresh_set.split(" ")

        #final optimal threshold sliced by hybcycle
        opt_thresh = float(opt_thresh_set[HybCycle])

        #get fwhm
        gen_path = img_parent / "threshold_counts" / f"Channel_{channel}"/ f"HybCycle_{HybCycle}" / f"MMStack_Pos{pos}"/ f"optimal_threshold_HybCycle_{HybCycle}.txt"
        fwhm = pd.read_csv(gen_path, sep="\t", header = None)
        fwhm_list = fwhm[0][0].split(" ")
        fwhm = float(fwhm_list[7])
            
        #read image
        img = tf.imread(img_src)
        
        #using daostarfinder detection
        if len(img.shape)==3:
            z=0
            peaks = daofinder(img[channel-1], threshold=opt_thresh,fwhm=fwhm)
            peaks = peaks.to_pandas()
            peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
            ch = np.zeros(len(peaks))+channel
            z_slice = np.zeros(len(peaks))+z
            peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
            peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
            dots = peaks
        else:
            dots = []
            for z in range(img.shape[0]):
                peaks = daofinder(img[z][channel-1], threshold=opt_thresh, fwhm=fwhm)
                peaks = peaks.to_pandas()
                peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
                ch = np.zeros(len(peaks))+channel
                z_slice = np.zeros(len(peaks))+z
                peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
                peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
                dots.append(peaks)
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
        #write out plot
        output_path_adj = output_path.replace(".csv", f"_{opt_thresh}.csv")
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
            plt.savefig(str(output_folder) + f"/size_hist_{opt_thresh}.png", dpi = 300)
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
            plt.savefig(str(output_folder) + f"/size_hist_{opt_thresh}.png", dpi = 300)
                       
        plt.show()
        plt.clf()
                        
        #write out final filtered dots
        if output == True:
            dots.to_csv(output_path)
        else:
            return dots
 
def dot_detection_parallel(img_src, HybCycle=0, size_cutoff=3, 
                           channel=1, pos_list=None, choose_thresh_set = 0, optimize=False, output=True):
    """
    This function will run dot detection in parallel, provided a list of images.
    
    Parameters
    ----------
    img_src= path to images
    HybCycle = which hybcycle the image belongs to
    size_cutoff = number of standard deviation away from mean size area
    channel = which channel to look at (1-4)
    pos_list = list of position numbers
    choose_thresh_set = int for which threshold set you want to use
    optimize = bool to test different threshold and min dots
    output = bool to output files
    
    Returns
    -------
    locations csv file and size distribution plots
    """
    
    import time
    start = time.time()
    
    if (optimize == True) and (type(img_src) != list):
            #image parent directory
            img_parent = Path(img_src).parent.parent.parent

            #read thresholds for each pos 
            opt_thresh_list = []
            for i in pos_list:
                gen_path = img_parent / "threshold_counts" / f"Channel_{channel}"/ f"HybCycle_{HybCycle}"/ f"MMStack_Pos{i}"/f"optimal_threshold_HybCycle_{HybCycle}.txt"
                opt_thresh_df = pd.read_csv(str(gen_path), sep="\t", header = None)
                opt_thresh_list.append(opt_thresh_df)

            #extract threshold values
            pos_thresh = []
            for p in range(len(opt_thresh_list)):
                p_list = []
                for e in range(len(opt_thresh_list[p])):
                    opt_thresh_split = opt_thresh_list[p][0][e].split(" ")
                    opt_thresh = float(opt_thresh_split[1])
                    fwhm = float(opt_thresh_split[7])
                    p_list.append(opt_thresh)
                pos_thresh.append(p_list)

            #generate thresh x pos array
            thresh_set = np.column_stack(pos_thresh)
            #get median
            thresh_median = np.median(thresh_set, axis=1) 
                
            with ProcessPoolExecutor(max_workers=32) as exe:
                futures = {}
                pos = pos_list[0]
                for opt_thresh in thresh_median:
                    fut = exe.submit(dot_detection, img_src, fwhm, HybCycle, size_cutoff,
                                     opt_thresh,channel,pos,choose_thresh_set,
                                     optimize, output)
                    futures[fut] = opt_thresh

                for fut in as_completed(futures):
                    opt_thresh = futures[fut]
                    print(f'Threshold {opt_thresh} completed after {(time.time() - start)/60} minutes')
    else:
        with ProcessPoolExecutor(max_workers=32) as exe:
            futures = {}
            #will get optimum threshold automatially
            opt_thresh = None
            pos = pos_list[0]
            for img in img_src:
                #get hybcycle number
                img_parent_cycle = Path(img).parent.name
                HybCycle_mod = int(img_parent_cycle.split("_")[1])
                #will get fwhm automatically so set to None
                fwhm = None
                #dot detect
                fut = exe.submit(dot_detection, img, fwhm, HybCycle_mod, size_cutoff,
                                 opt_thresh,channel,pos,choose_thresh_set,
                                 optimize, output)
                futures[fut] = img

            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {(time.time() - start)/60} seconds')
