#data management
import os
from pathlib import Path
import glob
from webfish_tools.util import find_matching_files
#image analysis
import tifffile as tf
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from photutils.centroids import centroid_2dg
#general analysis
import numpy as np
import pandas as pd
from scipy.stats import norm
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#denoising package
from scipy.signal import savgol_filter
#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")

def combine_dot_files(path_dots, hyb_start=0,hyb_end=64,num_HybCycle=32, pos= 0, 
                      channel = 0, num_z= 1, opt_files = False):
    """Function to read in all dot files and make combined locations csv
    Parameters
    ----------
    path_dots = path to dots_detected folder
    num_HybCYcle= number of total hyb cycles
    pos = position number
    channel = specific channel
    opt_files = get optimization files
    """
    if opt_files == True:
        #get paths for thresholded csvs
        dot_path_list = []
        for i in np.arange(hyb_start, hyb_end+1,1):
            dots_folders= Path(path_dots) / f"HybCycle_{i}"
            dot_files = list(dots_folders.glob(f'MMStack_Pos{pos}_*.csv'))
            dot_path_list.append(sorted(dot_files))
        #if num of thresholded csv is <11 then duplicate the lowest threshold
        for i in range(num_HybCycle):
            off = 11 - len(dot_path_list[i])
            if off == 0:
                continue
            else:
                for _ in range(off):
                    dot_path_list[i].insert(0,dot_path_list[i][0])
               
        #reorganize dot paths so that the lowest to highest threshold is matched for each hyb           
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
            comb_ch = comb[comb["ch"]==channel+1]
            for z in range(num_z):
                comb_z = comb_ch[comb_ch["z"]==z]
                output_folder = dot_path_list[0][0].parent.parent.parent.parent / "dots_comb"
                output_path = output_folder  /f"Channel_{channel+1}"/f"MMStack_Pos{pos}"/f"Threshold_{i}"/"Dot_Locations"
                output_path.mkdir(parents=True, exist_ok=True)
                comb_z.to_csv(str(output_path) + f"/locations_z_{z}.csv", index=False)
               
    else:
        #specify position
        position_dots = f'MMStack_Pos{pos}.csv'
        #get all paths for specific position across hybs
        files_dots, _, _ = find_matching_files(dot_paths, 'HybCycle_{hyb}' + f'/{position_dots}')
        files_dots = [str(f) for f in files_dots]
        
        dots = [pd.read_csv(j, index_col=0) for j in files_dots]
        con = pd.concat(dots)

        con_ch = con[con["ch"]==channel+1]
        for z in range(num_z):
            con_z = con_ch[con_ch["z"]==z]
            output_folder = dot_path_list[0][0].parent.parent.parent / "dots_comb"
            output_path = output_folder /f"Channel_{channel+1}"/f"MMStack_Pos{pos}"/"Dot_Locations"
            output_path.mkdir(parents=True, exist_ok=True)
            con_z.to_csv(str(output_path) +"/locations_z_{z}.csv", index=False)
                
def dot_counts(img_src, min_sigma = 1.5, max_sigma = 5, num_sigma = 5, threshold = 0.001, channel=0):
    """A function to count the number of dots found at a certain threshold
    Parameters
    ----------
    img_src= path to image
    min_sigma = minimum sigma for blob log
    max_sigma = maximum sigma for blob log
    num_sigma = number of intermediate sigmas
    threshold = threshold for blob log
    interval = spacing between maximum and minimum
    HybCycle= which hybcycle the image belongs to
    channel = which channel to look at
    
    Returns
    -------
    textfile containing counts detected at a certain threshold
    """
    
    #make output folder
    img_parent = Path(img_src).parent.parent.parent
    hyb_cycle = Path(img_src).parent.name
    pos_name = str(Path(img_src).name).replace(".ome.tif","") 
    output_path = Path(img_parent) / "threshold_counts" / f"Channel_{channel+1}" / hyb_cycle / pos_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    img = tf.imread(img_src)
    
    #blob detection across z
    counts = 0
    if len(img.shape)== 3:
        blobs_log = blob_log(img[channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                 num_sigma=num_sigma, threshold=threshold)
        counts += len(blobs_log)
        _list = [threshold, counts]
    else: 
        for z in range(img.shape[0]):
            blobs_log = blob_log(img[z][channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                 num_sigma=num_sigma, threshold=threshold)
            counts += len(blobs_log)
        _list = [threshold, counts]
    
    #make textfile
    f = open(str(output_path) + f"/thresh{threshold}_counts.txt" ,"w+")
    
    #write textfile
    for element in _list:
        f.write(str(element) + "\n")
    f.close()
    
def find_threshold(img_src, min_sigma = 1.5, max_sigma = 5, 
                   num_sigma = 5, threshold_min = 0.001, 
                   threshold_max= 0.1, interval=100, 
                   HybCycle = 0, channel = 0, min_dots_start=5000, 
                   min_dots_end = 11000,pos_start=5,pos_end=10, strict=False):
    
    """This function will find the optimal threshold to use during blob log detection
    Parameters
    ----------
    img_src= path to image
    min_sigma = minimum sigma for blob log
    max_sigma = maximum sigma for blob log
    num_sigma = number of intermediate sigmas
    threshold_min = start of threshold
    threshold_max = end of threshold
    interval = spacing between maximum and minimum
    HybCycle= which hybcycle the image belongs to
    channel = which channel to look at
    min_dots_start= minimum number of dots to start iteration
    min_dots_end= minimum number of dots to end iteration
    strict = if true it adjust threshold to be more stringent based on min_dots
    
    Returns
    ----------
    optimum threshold text file
    percent difference plot
    second derivative plot
    number of dots detected plot
    combined csv containing number of dots found at each threshold
    """
    
    #generate list of thresholds
    thresh_list = np.linspace(threshold_min,threshold_max, num = interval)
    thresh_list = list(thresh_list.flatten())
    
    #generate list of minimum dots
    min_dots = np.linspace(min_dots_start,min_dots_end,10).astype(int)
    
    import time
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=20) as exe:
        futures = {}
        if type(img_src) == list:
            for image in img_src:
                for thresh in thresh_list:
                    fut = exe.submit(dot_counts, image, min_sigma, max_sigma, num_sigma, thresh, channel)
                futures[fut] = image
                
            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {time.time() - start} seconds')
                
            print("completed tasks = ", len(futures))
        else:
            for thresh in thresh_list:
                fut = exe.submit(dot_counts, img_src, min_sigma, max_sigma, num_sigma, thresh, channel)
                futures[fut] = thresh

            for fut in as_completed(futures):
                thresh = futures[fut]
                print(f'Threshold {thresh} completed after {time.time() - start} seconds')
            print("completed tasks = ", len(futures))
    for _ in np.arange(pos_start,pos_end,1):
        #read in textfiles
        if type(img_src)!=list:
            img_parent = Path(img_src).parent.parent.parent
        else:
            img_parent = Path(img_src[0]).parent.parent.parent
        output_path = Path(img_parent) / "threshold_counts" /f"Channel_{channel+1}" / f"HybCycle_{HybCycle}" /f"MMStack_Pos{_}"
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

        #smooth data using Savitzky-Golay for better derivative computation
        #window of 15 and polynomial of 2
        smoothed_percent_change = savgol_filter(percent_change,15,2)

        # Compute second derivative of percent change
        smoothed_d2 = np.gradient(np.gradient(smoothed_percent_change))

        #find inflection points
        infls = np.argwhere(np.diff(np.sign(smoothed_d2)))

        #get threshold at inflection
        idx = infls[0][0]
        thresh = final["Threshold"].iloc[idx]

        #get counts at threshold
        counts_at_thresh = final["Total Counts"].iloc[idx]

        if strict == False:
            #check if threshold is above minimum counts expected
            thresh_counts = []
            #keep intitial values
            thresh_counts.append([thresh, counts_at_thresh, "initial"])
            for i in min_dots:
                if counts_at_thresh < i:
                    try:
                        new_idx = np.argwhere(final["Total Counts"].values > i)
                        new_idx = new_idx[len(new_idx)-1][0]
                        thresh_new = final["Threshold"].iloc[new_idx]
                        counts_at_thresh_new = final["Total Counts"].iloc[new_idx]  
                        thresh_counts.append([thresh_new, counts_at_thresh_new, i])
                    except IndexError:
                        continue
                else:
                    thresh_counts.append([thresh, counts_at_thresh, i])
        else:
            #check if threshold is below minimum counts expected
            thresh_counts = []
            #keep intitial values
            thresh_counts.append([thresh, counts_at_thresh, "initial"])
            for i in min_dots:
                if counts_at_thresh > i:
                    try:
                        new_idx = np.argwhere(final["Total Counts"].values < i)
                        new_idx = new_idx[0][0]
                        thresh_new = final["Threshold"].iloc[new_idx]
                        counts_at_thresh_new = final["Total Counts"].iloc[new_idx]  
                        thresh_counts.append([thresh_new, counts_at_thresh_new, i])
                    except IndexError:
                        continue
                else:
                    thresh_counts.append([thresh, counts_at_thresh, i])

        #plot percent difference        
        plt.plot(smoothed_percent_change)
        plt.xlabel("Each Threshold", size=12)
        plt.ylabel("Smoothed Percent Change", size=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        for i in range(len(thresh_counts)):
            plt_idx = final[final["Threshold"] == thresh_counts[i][0]].index
            plt.axhline(percent_change[plt_idx[0]], linestyle= "--",linewidth = 1, c = "red")
        sns.despine()
        plt.savefig(str(output_path) + f"/percent_diff_HybCycle_{HybCycle}", dpi = 300)
        plt.show()
        plt.clf()

        #plot first derivative        
        plt.plot(np.gradient(smoothed_percent_change))
        plt.xlabel("Each Threshold", size=12)
        plt.ylabel("First Derivative of Percent Difference", size=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        sns.despine()
        plt.savefig(str(output_path) + f"/first_deriv_percent_diff_HybCycle_{HybCycle}", dpi = 300)
        plt.show()
        plt.clf()

        #plot second derivative        
        plt.plot(smoothed_d2)
        plt.xlabel("Each Threshold", size=12)
        plt.ylabel("Second Derivative of Percent Difference", size=12)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        for i in range(len(infls)):
            plt.axvline(infls[i][0], linestyle= "--",linewidth = 1, c = "red")
        sns.despine()
        plt.savefig(str(output_path) + f"/second_deriv_percent_diff_HybCycle_{HybCycle}", dpi = 300)
        plt.show()
        plt.clf()

        #save result
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

        #make textfile
        f = open(str(output_path) + f"/optimal_threshold_HybCycle_{HybCycle}.txt" ,"w+")
        for element in thresh_counts:
            f.write(f"Threshold {element[0]} Counts {element[1]} Min_dots {element[2]}" + "\n")
        f.close()
    
def dot_detection(img_src, min_sigma = 1.5, max_sigma = 5, 
                  num_sigma = 5, HybCycle=0, size_cutoff=3, 
                  opt_thresh=0.001,channel=0,pos=0,choose_thresh_set = 0,hyb_number=64,check_initial = True,
                  gaussian = True, both = False, optimize=False, output=False):
    
    """This function use blob log dot detection using threshold obtained from find_threshold(). If gaussian is 
    set to true, then we centralize the dot after blob log by fitting a 2d gaussian.
    
    Parameters
    ----------
    img_src= path to image
    min_sigma = minimum sigma for blob log
    max_sigma = maximum sigma for blob log
    num_sigma = number of intermediate sigmas
    HybCycle= which hybcycle the image belongs to
    size_cutoff= number of standard deviation away from mean size area
    opt_thresh = threshold used during optimization
    channel = which channel to look at
    pos = position number for check initial
    choose_thresh = int for which threshold set you want to use
    check_initial = check initial optimal threshold
    hyb_number=total number of hybs for choose thresh set
    gaussian = bool for gaussian fitting
    both =  bool to return both log and gaussian fitted locations
    optimize = bool to test different threshold and min dots
    ouput = bool to output files
    
    Returns
    ----------
    MMStack_Pos{x}.csv
    """             
    img_parent = Path(img_src).parent.parent.parent
    output_folder = Path(img_parent) / "dots_detected"/ f"Channel_{channel+1}" / f"HybCycle_{HybCycle}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = str(output_folder / Path(img_src).name)
    output_path = output_path.replace(".ome.tif",".csv")
    #read image
    img = tf.imread(img_src)
    #get optimal threshold
    if optimize == True:
        #using blob log dot detection
        if len(img.shape)==3:
            z=0
            blobs_log = blob_log(img[channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                         num_sigma=num_sigma, threshold=opt_thresh)
            blobs_log = blobs_log[:,0:2]
            ch = np.zeros(len(blobs_log))+channel+1
            z_slice = np.zeros(len(blobs_log))+z
            blobs_log = np.append(blobs_log, ch.reshape(len(ch),1), axis=1)
            blobs_log = np.append(blobs_log, z_slice.reshape(len(z_slice),1), axis=1)
            dots = blobs_log
        else:
            dots = []
            for z in range(img.shape[0]):
                blobs_log = blob_log(img[z][channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                         num_sigma=num_sigma, threshold=opt_thresh)
                blobs_log = blobs_log[:,0:2]
                ch = np.zeros(len(blobs_log))+channel+1
                z_slice = np.zeros(len(blobs_log))+z
                blobs_log = np.append(blobs_log, ch.reshape(len(ch),1), axis=1)
                blobs_log = np.append(blobs_log, z_slice.reshape(len(z_slice),1), axis=1)
                dots.append(blobs_log)
            dots = np.concatenate(dots)
            
        #make df and reorganize        
        dots = pd.DataFrame(dots)
        dots.columns = ["y", "x", "ch", "z"]
        dots["hyb"] = HybCycle
        dots = dots[["hyb","ch","x","y","z"]]

        #get intensities
        #subtract 1 from channels to get right slice
        #added 1 above to make sure it aligns with decoding format
        coord = dots[["x","y","z","ch"]].values
        int_list = []
        area_list = []
        for i in coord:
            x = int(i[0])
            y = int(i[1])
            z = int(i[2])
            c = int(i[3])
            #get intensity
            if len(img.shape)==3:
                intensity = img[c-1][y,x]
            else:
                intensity = img[z][c-1][y,x]
            #-3 and +3 is to get a larger window for dot
            try:
                if len(img.shape)==3:
                    blob = img[c-1][y-3:y+3,x-3:x+3]
                else:
                    blob = img[z][c-1][y-3:y+3,x-3:x+3]
            except IndexError:
                int_list.append(intensity)
                area_list.append(0)
            #get area
            try:
                otsu = threshold_otsu(blob)
                label_otsu = label(blob >= otsu)
                area = regionprops(label_otsu)[0]["area"]
                int_list.append(intensity)
                area_list.append(area)
            except ValueError:
                int_list.append(intensity)
                area_list.append(0)
            except IndexError:
                int_list.append(intensity)
                area_list.append(0)
  
        dots["intensity"] = int_list
        dots["area"] = area_list

        if gaussian == True:
            gaussian_fits = []
            for i in coord:
                x = int(i[0])
                y = int(i[1])
                z = int(i[2])
                c = int(i[3])
                #just get intensity
                if len(img.shape)==3:
                    _int = img[c-1][y,x]
                else:
                    _int = img[z][c-1][y,x]
                #-3 and +3 is to get a larger window for dot
                try:
                    if len(img.shape)==3:
                        blob = img[c-1][y-3:y+3,x-3:x+3]
                    else:
                        blob = img[z][c-1][y-3:y+3,x-3:x+3]
                except IndexError:
                    continue
                #get area estimate
                try:
                    otsu = threshold_otsu(blob)
                    label_otsu = label(blob >= otsu)
                    area = regionprops(label_otsu)[0]["area"]
                except ValueError:
                    area = 0
                except IndexError:
                    area = 0
                #fitting gaussian using photutils package
                try:
                    x_g, y_g = centroid_2dg(blob)
                    y_offset = np.abs(y_g-3)
                    x_offset = np.abs(x_g-3)
                    #apply offset to dots
                    if y_g > 3:
                        y = y+y_offset
                    else:
                        y = y-y_offset
                    if x_g > 3:
                        x = x+x_offset
                    else:
                        x = x-x_offset
                    gaussian_fits.append([x,y,z,c, _int, area])
                except ValueError:
                    continue
            #make new df
            gaussian_fit = pd.DataFrame(gaussian_fits)
            gaussian_fit.columns = ["x", "y","z","ch","intensity", "area"]
            gaussian_fit["hyb"] = HybCycle
            gaussian_fit = gaussian_fit[["hyb","ch","x","y","z","intensity", "area"]]
            #write out results
            output_path_adj = output_path.replace(".csv", f"_{opt_thresh}.csv")

            #filter by size
            mu, std = norm.fit(gaussian_fit["area"]) #fit gaussian to size dataset
            plt.hist(gaussian_fit["area"], density=True, bins=20)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x,p, label="Gaussian Fitted Data")
            plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
            plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
            plt.xlabel("Area by pixel")
            plt.ylabel("Proportion")
            plt.savefig(str(output_folder) + f"/size_hist_{opt_thresh}.png", dpi = 300)
            plt.show()
            plt.clf()
            gaussian_fit = gaussian_fit[(gaussian_fit["area"] < (mu+(size_cutoff*std))) 
                                        & (gaussian_fit["area"] > (mu-(size_cutoff*std)))]

            gaussian_fit.to_csv(output_path_adj)
        else: 
            #filter by size
            mu, std = norm.fit(dots["area"]) #fit gaussian to size dataset
            plt.hist(dots["area"], density=True, bins=20)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x,p, label="Gaussian Fitted Data")
            plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
            plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
            plt.xlabel("Area by pixel")
            plt.ylabel("Proportion")
            plt.savefig(str(output_folder) + f"/size_hist_{opt_thresh}.png", dpi = 300)
            plt.show()
            plt.clf()
            dots = dots[(dots["area"] < (mu+(size_cutoff*std))) 
                                        & (dots["area"] > (mu-(size_cutoff*std)))]

            output_path_adj = output_path.replace(".csv", f"_{opt_thresh}.csv")
            dots.to_csv(output_path_adj)
    else:
        #output folder
        img_parent = Path(img_src).parent.parent.parent
        output_folder = Path(img_parent) / "dots_detected" /"final" /f"Channel_{channel+1}"/f"HybCycle_{HybCycle}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = str(output_folder / Path(img_src).name)
        output_path = output_path.replace(".ome.tif",f"_{channel}.csv")
        
        #get threshold
        if check_initial == True:
            #take initial threshold parameter
            gen_path = img_parent / "threshold_counts" / f"Channel_{channel+1}"/ f"HybCycle_{HybCycle}" / f"MMStack_Pos{pos}"/ f"optimal_threshold_HybCycle_{HybCycle}.txt"
            opt_thresh = pd.read_csv(gen_path, sep="\t", header = None)
            opt_thresh = opt_thresh[0][0].split(" ")
            opt_thresh = float(opt_thresh[1])
        else:
            #get optimal threshold after decoding verification
            gen_path = img_parent / "dots_detected" /f"Channel_{channel+1}" /f"optimal_threshold_test_complete_ch{channel}.txt"
            opt_thresh = pd.read_csv(gen_path, sep="\t", header = None)
            #pick thresh set
            opt_thresh_set = opt_thresh[0][choose_thresh_set]

            #parse list of thresholds and pick correct one for each hyb
            opt_thresh_set = opt_thresh_set.split(" ")

            #final optimal threshold sliced by hybcycle
            opt_thresh = float(opt_thresh_set[HybCycle])
        
        #read image
        img = tf.imread(img_src)
        
        #using blob log dot detection
        if len(img.shape) == 3:
            z=0
            blobs_log = blob_log(img[channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                     num_sigma=num_sigma, threshold=opt_thresh)
            blobs_log = blobs_log[:,0:2]
            ch = np.zeros(len(blobs_log))+channel+1
            z_slice = np.zeros(len(blobs_log))+z
            blobs_log = np.append(blobs_log, ch.reshape(len(ch),1), axis=1)
            blobs_log = np.append(blobs_log, z_slice.reshape(len(z_slice),1), axis=1)
            dots = blobs_log
        else:
            dots = []
            for z in range(img.shape[0]):
                blobs_log = blob_log(img[z][channel], min_sigma = min_sigma, max_sigma=max_sigma, 
                                     num_sigma=num_sigma, threshold=opt_thresh)
                blobs_log = blobs_log[:,0:2]
                ch = np.zeros(len(blobs_log))+channel+1
                z_slice = np.zeros(len(blobs_log))+z
                blobs_log = np.append(blobs_log, ch.reshape(len(ch),1), axis=1)
                blobs_log = np.append(blobs_log, z_slice.reshape(len(z_slice),1), axis=1)
                dots.append(blobs_log)
            dots = np.concatenate(dots)

        #make df and reorganize        
        dots = pd.DataFrame(dots)
        dots.columns = ["y", "x", "ch", "z"]
        dots["hyb"] = HybCycle
        dots = dots[["hyb","ch","x","y","z"]]

        ##load raw image
        #img_raw = tf.imread(raw_src)

        #get intensities
        #subtract 1 from channels to get right slice
        #added 1 above to make sure it aligns with decoding format
        coord = dots[["x","y","z","ch"]].values
        int_list = []
        area_list = []
        for i in coord:
            x = int(i[0])
            y = int(i[1])
            z = int(i[2])
            c = int(i[3])
            #-3 and +3 is to get a larger window for dot
            if len(img.shape)==3:
                blob = img[c-1][y-3:y+3,x-3:x+3]
            else:
                blob = img[z][c-1][y-3:y+3,x-3:x+3]
            #get intensity
            if len(img.shape)==3:
                intensity = img[c-1][y,x]
            else:
                intensity = img[z][c-1][y,x]
            #get area estimate
            try:
                otsu = threshold_otsu(blob)
                label_otsu = label(blob >= otsu)
                area = regionprops(label_otsu)[0]["area"]
                int_list.append(intensity)
                area_list.append(area)
            except ValueError:
                int_list.append(intensity)
                area_list.append(0)
            except IndexError:
                int_list.append(intensity)
                area_list.append(0)

        dots["intensity"] = int_list
        dots["area"] = area_list

        if gaussian == True:
            gaussian_fits = []
            for i in coord:
                x = int(i[0])
                y = int(i[1])
                z = int(i[2])
                c = int(i[3])
                #-3 and +3 is to get a larger window for dot
                if len(img.shape) ==3:
                    blob = img[c-1][y-3:y+3,x-3:x+3]
                else:
                    blob = img[z][c-1][y-3:y+3,x-3:x+3]
                #just get intensity
                if len(img.shape)==3:
                    _int = img[c-1][y,x]
                else:
                    _int = img[z][c-1][y,x]
                #get area estimate
                try:
                    otsu = threshold_otsu(blob)
                    label_otsu = label(blob>=otsu)
                    area = regionprops(label_otsu)[0]["area"]
                except ValueError:
                    area= 0
                except IndexError:
                    area=0
                #fitting gaussian using photutils package
                try:
                    x_g, y_g = centroid_2dg(blob)
                    y_offset = np.abs(y_g-3)
                    x_offset = np.abs(x_g-3)
                    #apply offset to dots
                    if y_g > 3:
                        y = y+y_offset
                    else:
                        y = y-y_offset
                    if x_g > 3:
                        x = x+x_offset
                    else:
                        x = x-x_offset
                    gaussian_fits.append([x,y,z,c, _int,area])
                except ValueError:
                    continue
                except IndexError:
                    continue
            #make new df
            gaussian_fit = pd.DataFrame(gaussian_fits)
            gaussian_fit.columns = ["x", "y","z","ch","intensity","area"]
            gaussian_fit["hyb"] = HybCycle
            gaussian_fit = gaussian_fit[["hyb","ch","x","y","z","intensity","area"]]

            if both == True:
                return gaussian_fit, dots
            else:
                #filter by size
                mu, std = norm.fit(gaussian_fit["area"]) #fit gaussian to size dataset
                plt.hist(gaussian_fit["area"], density=True, bins=20)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x,p, label="Gaussian Fitted Data")
                plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
                plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
                plt.xlabel("Area by pixel")
                plt.ylabel("Proportion")
                plt.legend()
                if output == True:
                    plt.savefig(str(output_folder) + f"/size_hist.png", dpi = 300)
                plt.show()
                plt.clf()
                gaussian_fit = gaussian_fit[(gaussian_fit["area"] < (mu+(size_cutoff*std))) 
                                            & (gaussian_fit["area"] > (mu-(size_cutoff*std)))]
                if output == True:
                    gaussian_fit.to_csv(output_path)
                return gaussian_fit
        else: 
            #print(dots["area"].dropna().dtype)
            #filter by size
            mu, std = norm.fit(dots["area"].dropna()) #fit gaussian to size dataset
            plt.hist(dots["area"], density=True, bins=20)
            plt.show()
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x,p, label="Gaussian Fitted Data")
            plt.axvline(mu+(size_cutoff*std), ls="--", c = "red")
            plt.axvline(mu-(size_cutoff*std), ls="--",c = "red")
            plt.xlabel("Area by pixel")
            plt.ylabel("Proportion")
            plt.legend()
            if output == True:
                plt.savefig(str(output_folder) + f"/size_hist.png", dpi = 300)
            plt.show()
            plt.clf()
            dots = dots[(dots["area"] < (mu+(size_cutoff*std))) 
                                        & (dots["area"] > (mu-(size_cutoff*std)))]
               
            if output == True:
                dots.to_csv(output_path)
            return dots

    
def dot_detection_parallel(img_src, min_sigma = 1.5, max_sigma = 5, 
                           num_sigma = 5, HybCycle=0, size_cutoff=3, 
                           channel=0, pos_start=0,pos_end=10,choose_thresh_set = 0,hyb_number=64,
                           check_initial = False,gaussian = True, 
                           both = False, optimize=False, output=True):
    """Find dots for all positions and hyb"""
    
    import time
    start = time.time()
    
    if (optimize == True) and (type(img_src) != list):
            #image parent directory
            img_parent = Path(img_src).parent.parent.parent

            #read thresholds for each pos 
            opt_thresh_list = []
            for i in np.arange(pos_start,pos_end,1):
                gen_path = img_parent / "threshold_counts" / f"Channel_{channel+1}"/ f"HybCycle_{HybCycle}"/ f"MMStack_Pos{i}"/f"optimal_threshold_HybCycle_{HybCycle}.txt"
                opt_thresh_df = pd.read_csv(str(gen_path), sep="\t", header = None)
                opt_thresh_list.append(opt_thresh_df)

            #extract threshold values
            pos_thresh = []
            for p in range(len(opt_thresh_list)):
                for e in range(len(opt_thresh_list[p])):
                    opt_thresh_split = opt_thresh_list[p][0][e].split(" ")
                    opt_thresh = float(opt_thresh_split[1])
                    try:
                        min_dots = int(opt_thresh_split[5])
                    except ValueError:
                        min_dots = "initial"
                    pos_thresh.append([opt_thresh, min_dots])

            #get median threshold per min dot parameter
            convert_df = pd.DataFrame(pos_thresh)
            thresh_median = []
            for min_dot in convert_df[1].unique():
                slice_by_mindot = convert_df[convert_df[1]==min_dot]
                thresh_median.append(np.median(slice_by_mindot[0]))  
     
            with ProcessPoolExecutor(max_workers=40) as exe:
                futures = {}
                pos = None
                for opt_thresh in thresh_median:
                    fut = exe.submit(dot_detection, img_src, min_sigma, max_sigma, 
                                         num_sigma, HybCycle, size_cutoff,
                                         opt_thresh,channel,pos,choose_thresh_set,hyb_number,check_initial,
                                         gaussian, both, optimize, output)
                    futures[fut] = opt_thresh

                for fut in as_completed(futures):
                    opt_thresh = futures[fut]
                    print(f'Threshold{opt_thresh} completed after {time.time() - start} seconds')

                print("completed thresholds = ", len(futures))

    else:
        with ProcessPoolExecutor(max_workers=24) as exe:
            futures = {}
            opt_thresh = None
            pos=None
            for img in img_src:
                #get hybcycle number
                img_parent_cycle = Path(img).parent.name
                HybCycle_mod = int(img_parent_cycle.split("_")[1])
                #dot detect
                fut = exe.submit(dot_detection, img, min_sigma, max_sigma, 
                                 num_sigma, HybCycle_mod, size_cutoff,
                                 opt_thresh,channel,pos,choose_thresh_set,hyb_number,check_initial,
                                 gaussian, both, optimize, output)
                futures[fut] = img

            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {time.time() - start} seconds')

            print("completed jobs = ", len(futures))