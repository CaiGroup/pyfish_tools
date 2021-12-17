#data management
import os
from pathlib import Path
import glob
from webfish_tools.util import find_matching_files
from sys import exit
#image analysis
import tifffile as tf
from photutils.detection import DAOStarFinder
from skimage.filters import threshold_otsu
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

def dot_recognition_optimize(data,  fwhm=2.4, threshold=100,sigma_radius=2, 
                             brightest=None, roundlo=0.5,roundhi=1, z=0):
    ## Input: a matrix object representing an image
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, sigma_radius=sigma_radius, 
                            brightest=brightest,roundlo=roundlo,roundhi=roundhi)
    #print("std: "+ str(std))
    sources = daofind(data)
    if sources == None:
        return None
    for col in sources.colnames:
         sources[col].info.format = '%.8g'  # for consistent table output
    df = sources["xcentroid","ycentroid", "peak", "npix"].to_pandas()
    df.columns = ["x","y","intensity","area"]
    df["z"] = z
    return df
                
def dot_counts(img_src, fwhm=2.4, threshold=100,sigma_radius=2, brightest=None, roundlo=0.5,roundhi=1):
    
    #make output folder
    img_parent = Path(img_src).parent.parent.parent
    hyb_cycle = Path(img_src).parent.name
    pos_name = str(Path(img_src).name).replace(".ome.tif","") 
    output_path = Path(img_parent) / "threshold_counts" / hyb_cycle / pos_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    img = tf.imread(img_src)
    
    #if channel is all
    #blob detection across z and channels
    if channel == "all":
        
        counts = 0
        if len(img.shape)== 3:
            for c in range(img.shape[0]):
                dots = dot_recognition_optimize(img[c], fwhm=fwhm, threshold=threshold, sigma_radius=sigma_radius, 
                             brightest=brightest, roundlo=roundlo,roundhi=roundhi, z=0)
                counts += len(dots)
                #make list
                _list = [threshold, counts, c]

                #make textfile
                f = open(str(output_path) + f"/thresh{threshold}_channel{c}_counts.txt" ,"w+")

                #write textfile
                for element in _list:
                    f.write(str(element) + "\n")
                f.close()
                #reset counts
                counts = 0
        else:
            for c in range(img.shape[1]):
                for z in range(img.shape[0]):
                    dots = dot_recognition_optimize(img[z][c], fwhm=fwhm, threshold=threshold, sigma_radius=sigma_radius, 
                             brightest=brightest, roundlo=roundlo,roundhi=roundhi, z=z)
                    counts += len(dots)
                #make list
                _list = [threshold, counts, c]

                #make textfile
                f = open(str(output_path) + f"/thresh{threshold}_channel{c}_counts.txt" ,"w+")

                #write textfile
                for element in _list:
                    f.write(str(element) + "\n")
                f.close()
                #reset counts
                counts = 0
    
def find_threshold(img_src,sigma_radius=2, roundlo=0.5,roundhi=1, brightest=None,
                   threshold_min = 100, 
                   threshold_max= 3000, interval=50, 
                   HybCycle = 0, channel = 0, min_dots_start=5000,
                   min_dots_end = 11000,num_pos=5, strict=False):
    
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
                    fut = exe.submit(dot_counts,image, fwhm, thresh, sigma_radius, brightest, roundlo,roundhi)
                futures[fut] = image

            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {time.time() - start} seconds')
                
            print("completed tasks = ", len(futures))
        else:
            for thresh in thresh_list:
                fut = exe.submit(dot_counts, img_src,  fwhm, threshold,sigma_radius, brightest, roundlo,roundhi)
                futures[fut] = thresh

            for fut in as_completed(futures):
                thresh = futures[fut]
                print(f'Threshold {thresh} completed after {time.time() - start} seconds')
                
            print("completed tasks = ", len(futures))
    for _ in range(num_pos):
        #read in textfiles
        if len(img_src)==1:
            img_parent = Path(img_src).parent.parent.parent
        else:
            img_parent = Path(img_src[0]).parent.parent.parent
        output_path = Path(img_parent) / "threshold_counts" / f"HybCycle_{HybCycle}" /f"MMStack_Pos{_}"
        paths = glob.glob(str(output_path) +"/*_counts.txt")

        data_list= []
        for i in paths:
            data = pd.read_csv(i, sep=" ", header=None).T
            data_list.append(data)

        #combine text files to get final csv
        final = pd.concat(data_list)
        final.columns = ["Threshold", "Total Counts", "Channel"]
        final = final.sort_values("Threshold")
        final = final.reset_index(drop=True)
        final.to_csv(str(output_path) + f"/combined_thresh_HybCycle{HybCycle}.csv")

        #remove text files
        files_all = os.listdir(str(output_path))
        for item in files_all:
            if item.endswith(".txt"):
                os.remove(os.path.join(str(output_path), item))

        #split df by channels
        df_ch = []
        for i in sorted(final["Channel"].unique()):
            df_ch.append((final[final["Channel"]==i]).reset_index(drop=True))
        #loop through channels
        for c in range(len(df_ch)):
            #calculate percent change
            vector_sum = df_ch[c]["Total Counts"].values[:-1] + df_ch[c]["Total Counts"].values[1:]
            percent_change = np.abs(np.diff(df_ch[c]["Total Counts"]))/(vector_sum/2)

            #smooth data using Savitzky-Golay for better derivative computation
            #window of 15 and polynomial of 2
            smoothed_percent_change = savgol_filter(percent_change,15,2)

            # Compute second derivative of percent change
            smoothed_d2 = np.gradient(np.gradient(smoothed_percent_change))

            #find inflection points
            infls = np.argwhere(np.diff(np.sign(smoothed_d2)))

            #get threshold at inflection
            idx = infls[0][0]
            thresh = df_ch[c]["Threshold"].iloc[idx]

            #get counts at threshold
            counts_at_thresh = df_ch[c]["Total Counts"].iloc[idx]

            if strict == False:
                #check if threshold is above minimum counts expected
                thresh_counts = []
                #keep intitial values
                thresh_counts.append([thresh, counts_at_thresh, "initial"])
                for i in min_dots:
                    if counts_at_thresh < i:
                        try:
                            new_idx = np.argwhere(df_ch[c]["Total Counts"].values > i)
                            new_idx = new_idx[len(new_idx)-1][0]
                            thresh_new = df_ch[c]["Threshold"].iloc[new_idx]
                            counts_at_thresh_new = df_ch[c]["Total Counts"].iloc[new_idx]  
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
                            new_idx = np.argwhere(df_ch[c]["Total Counts"].values < i)
                            new_idx = new_idx[0][0]
                            thresh_new = df_ch[c]["Threshold"].iloc[new_idx]
                            counts_at_thresh_new = df_ch[c]["Total Counts"].iloc[new_idx]  
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
                plt_idx = df_ch[c][df_ch[c]["Threshold"] == thresh_counts[i][0]].index
                plt.axhline(percent_change[plt_idx[0]-1], linestyle= "--",linewidth = 1, c = "red")
            sns.despine()
            plt.savefig(str(output_path) + f"/percent_diff_HybCycle_{HybCycle}_ch{c}", dpi = 300)
            plt.show()
            plt.clf()

            #plot first derivative        
            plt.plot(np.gradient(smoothed_percent_change))
            plt.xlabel("Each Threshold", size=12)
            plt.ylabel("First Derivative of Percent Difference", size=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            sns.despine()
            plt.savefig(str(output_path) + f"/first_deriv_percent_diff_HybCycle_{HybCycle}_ch{c}", dpi = 300)
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
            plt.savefig(str(output_path) + f"/second_deriv_percent_diff_HybCycle_{HybCycle}_ch{c}", dpi = 300)
            plt.show()
            plt.clf()

            #save result
            plt.plot(df_ch[c]["Threshold"], df_ch[c]["Total Counts"])
            plt.xlabel("Threshold", size=12)
            plt.ylabel("Total Counts", size=12)
            plt.xticks(fontsize=12, rotation=0)
            plt.yticks(fontsize=12, rotation=0)
            for i in range(len(thresh_counts)):
                plt.axvline(thresh_counts[i][0], linestyle= "--",linewidth = 1, c = "red")
            sns.despine()
            plt.savefig(str(output_path) + f"/totalcounts_HybCycle_{HybCycle}_ch{c}", dpi = 300)
            plt.show()
            plt.clf()

            #make textfile
            f = open(str(output_path) + f"/optimal_threshold_HybCycle_{HybCycle}_ch{c}.txt" ,"w+")
            for element in thresh_counts:
                f.write(f"Threshold {element[0]} Counts {element[1]} Min_dots {element[2]}" + "\n")
            f.close()
    
def dot_detection(img_src, min_sigma = 1.5, max_sigma = 5, 
                  num_sigma = 5, HybCycle=0, size_cutoff=3, 
                  opt_thresh=0.001,channel=0,pos=0,choose_thresh_set = 0,check_initial = True,
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
    opt_thresh = optimal threshold from find thresh (used during optimize ==True)
    channel = channel information
    pos = position information used for check initial set to True
    choose_thresh_set = pick threshold set from final list after decoding check
    check_initial = check initial starting point threshold
    gaussian = bool for gaussian fitting
    both =  bool to return both log and gaussian fitted locations
    optimize = bool to test different threshold values from find_threshold()
    output= bool for writing out files
    
    Returns
    ----------
    MMStack_Pos{x}.csv
    size_hist.png
    """             
   
    #get optimal thresholds
    if optimize == True:
        img_parent = Path(img_src).parent.parent.parent
        output_folder = Path(img_parent) / "dots_detected" / f"HybCycle_{HybCycle}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = str(output_folder / Path(img_src).name)
        output_path = output_path.replace(".ome.tif",".csv")
        
        #read image
        img = tf.imread(img_src)
        
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
        #remember that rows are ys and columns are xs
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
            if len(img.shape) == 3:
                intensity = img[c-1][y,x]
            else:
                intensity = img[z][c-1][y,x]
            #-3 and +3 is to get a larger window for dot
            if len(img.shape)==3:
                blob = img[c-1][y-3:y+3,x-3:x+3]
            else:
                blob = img[z][c-1][y-3:y+3,x-3:x+3]
            #get area
            try:
                otsu = threshold_otsu(blob)
                label_otsu = label(blob >= otsu)
                area = regionprops(label_otsu)[0]["area"]
                int_list.append(intensity)
                area_list.append(area)
            except ValueError:
                int_list.append(intensity)
                area_list.append("na")
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
                if len(img.shape) == 3:
                    _int = img[c-1][y,x]
                else:
                    _int = img[z][c-1][y,x]
                #-3 and +3 is to get a larger window for dot
                if len(img.shape) == 3:
                    blob = img[c-1][y-3:y+3,x-3:x+3]
                else:
                    blob = img[z][c-1][y-3:y+3,x-3:x+3]
                #get area estimate
                try:
                    otsu = threshold_otsu(blob)
                    label_otsu = label(blob >= otsu)
                    area = regionprops(label_otsu)[0]["area"]
                except ValueError:
                    area = "na"
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
            plt.savefig(str(output_folder) + f"/size_hist_Threshold{opt_thresh}_ch{channel}.png", dpi = 300)
            plt.clf()

            gaussian_fit = gaussian_fit[(gaussian_fit["area"] < (mu+(size_cutoff*std)))
                                        & (gaussian_fit["area"] > (mu-(size_cutoff*std)))]
            
            #write out results
            output_path_adj = output_path.replace(".csv", f"_Threshold{opt_thresh}_ch{channel}.csv")
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
            plt.legend()
            plt.savefig(str(output_folder) + f"/size_hist_Threshold{opt_thresh}_ch{channel}.png", dpi = 300)
            plt.clf()

            dots = dots[(dotst["area"] < (mu+(size_cutoff*std))) 
                                        & (dots["area"] > (mu-(size_cutoff*std)))]

            output_path_adj = output_path.replace(".csv", f"_Threshold{opt_thresh}_ch{channel}.csv")
            dots.to_csv(output_path_adj)
    else:
        #output folder
        img_parent = Path(img_src).parent.parent.parent
        output_folder = Path(img_parent) / "dots_detected" /"final" /f"HybCycle_{HybCycle}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = str(output_folder / Path(img_src).name)
        output_path = output_path.replace(".ome.tif",f"_{channel}.csv")
        
        #get threshold
        if check_initial == True:
            #take initial threshold parameter
            gen_path = img_parent / "threshold_counts" / f"HybCycle_{HybCycle}" / f"MMStack_Pos{pos}"/ f"optimal_threshold_HybCycle_{HybCycle}_ch{channel}.txt"
            opt_thresh = pd.read_csv(gen_path, sep="\t", header = None)
            opt_thresh = opt_thresh[0][0].split(" ")
            opt_thresh = float(opt_thresh[1])
        else:
            #get optimal threshold after decoding verification
            gen_path = img_parent / "dots_detected" / "optimal_threshold_test_complete.txt"
            opt_thresh = pd.read_csv(gen_path, sep="\t", header = None)
            opt_thresh_list = []
            k=0
            #split by number of total threshold sets
            #take every 48 indicies (4 channels times 12 hybs)
            for _ in range(11):
                opt_thresh_list.append(opt_thresh.iloc[k:k+48])
                k += 48

            opt_thresh_set = opt_thresh_list[choose_thresh_set]
            #make list of 4 arrays. Each array will be specific towards a channel.
            split_by_channel=[]
            k=0
            for _ in range(4):
                split_by_channel.append(opt_thresh_set.iloc[k:k+12].values)
                k+=12
            #final optimal threshold sliced by channel and hybcycle
            opt_thresh = float(split_by_channel[channel][HybCycle][0])
        
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
                area_list.append("na")

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
                    area= "na"
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
                           num_pos=5,num_channels=4, pos=0,choose_thresh_set = 0,
                           check_initial = False,gaussian = True, 
                           both = False, optimize=False, output=True):
                            
    """Find dots for all positions and hyb"""
    
    import time
    start = time.time()
    if (optimize == True) and (type(img_src) != list):
        #image parent directory
        img_parent = Path(img_src).parent.parent.parent
       
        #read thresholds per channel for each pos 
        opt_thresh_list = []
        for c in range(num_channels):
            channel_thresh = []
            for i in range(num_pos):
                gen_path = img_parent / "threshold_counts" / f"HybCycle_{HybCycle}"/ f"MMStack_Pos{i}"/f"optimal_threshold_HybCycle_{HybCycle}_ch{c}.txt"
                opt_thresh_df = pd.read_csv(str(gen_path), sep="\t", header = None)
                channel_thresh.append(opt_thresh_df)
            opt_thresh_list.append(channel_thresh)

        #extract threshold values
        channel_thresh = []
        for c in range(len(opt_thresh_list)):
            pos_thresh = []
            for p in range(num_pos):
                for e in range(len(opt_thresh_list[c][p])):
                    opt_thresh_split = opt_thresh_list[c][p][0][e].split(" ")
                    opt_thresh = float(opt_thresh_split[1])
                    try:
                        min_dots = int(opt_thresh_split[5])
                    except ValueError:
                        min_dots = "initial"
                    pos_thresh.append([opt_thresh, min_dots])
            channel_thresh.append(pos_thresh)

        #get median threshold per channel
        channel_thresh_final = []
        for i in range(len(channel_thresh)):
            convert_df = pd.DataFrame(channel_thresh[i])
            channel_thresh_median = []
            for min_dot in convert_df[1].unique():
                slice_by_mindot = convert_df[convert_df[1]==min_dot]
                channel_thresh_median.append(np.median(slice_by_mindot[0]))
            channel_thresh_final.append(channel_thresh_median)  
    
        with ProcessPoolExecutor(max_workers=40) as exe:
            futures = {}
            pos = None
            for c in range(len(channel_thresh_final)):
                for opt_thresh in channel_thresh_final[c]:
                    fut = exe.submit(dot_detection, img_src, min_sigma, max_sigma, 
                                     num_sigma, HybCycle, size_cutoff,
                                     opt_thresh,c,pos,choose_thresh_set,check_initial,
                                     gaussian, both, optimize, output)
                futures[fut] = c

            for fut in as_completed(futures):
                c = futures[fut]
                print(f'Channel {c} completed after {time.time() - start} seconds')

            print("completed channels = ", len(futures))

    else:
        with ProcessPoolExecutor(max_workers=24) as exe:
            futures = {}
            opt_thresh = None
            pos=None
            for img in img_src:
                #get hybcycle number
                img_parent_cycle = Path(img).parent.name
                HybCycle_mod = int(img_parent_cycle.split("_")[1])
                #loop through channels
                for c in range(num_channels):
                    fut = exe.submit(dot_detection, img, min_sigma, max_sigma, 
                                     num_sigma, HybCycle_mod, size_cutoff,
                                     opt_thresh,c,pos,choose_thresh_set,check_initial,
                                     gaussian, both, optimize, output)
                    futures[fut] = img

            for fut in as_completed(futures):
                img = futures[fut]
                print(f'{img} completed after {time.time() - start} seconds')

            print("completed jobs = ", len(futures))
            
def combine_dot_files(path_dots, num_HybCycle = 0, pos= 0, 
                      num_channels = 1, num_z= 1, opt_files = False):
    """Function to read in all dot files and make combined locations csv
    Parameters
    ----------
    path_dots = path to dots_detected folder
    num_HybCYcle= number of total hyb cycles
    pos = position number
    num_channels = number of channels
    opt_files = get optimization files
    """
    if opt_files == True:
        
        #get paths for thresholded csvs
        dot_path_list = []
        for ch in range(num_channels):
            for i in range(num_HybCycle):
                dots_folders= Path(path_dots) / f"HybCycle_{i}"
                dot_files = list(dots_folders.glob(f'MMStack_Pos{pos}_*_ch{ch}.csv'))
                dot_path_list.append(sorted(dot_files))

        #if num of thresholded csv is <11 then duplicate the lowest threshold
        for i in range(num_HybCycle*num_channels):
            off = 11 - len(dot_path_list[i])
            if off == 0:
                continue
            else:
                failed = []
                for _ in range(off):
                    try:
                        dot_path_list[i].insert(0,dot_path_list[i][0])
                    except IndexError:
                        if i < num_HybCycle:
                            print("missing HybCycle", i)
                            failed.append(i)
                        elif i > num_HybCycle and i <(num_HybCycle*2):
                            print("missing HybCycle", i-12)
                            failed.append(i)
                        elif i > (num_HybCycle*2) and i <(num_HybCycle*3):
                            print("missing HybCycle", i-24)
                            failed.append(i)
                        elif i > (num_HybCycle*3) and i <(num_HybCycle*4):
                            print("missing HybCycle", i-36)
                            failed.append(i)
                if failed != []:
                    print("files are missing, ending loop")
                    exit()


        #reorganize dot paths so that the lowest to highest threshold is matched for each hyb   
        #final reorganized should be list of 44. Every 4th interval is next set of thresholds.
        reorg_dot_files = []
        for i in range(11):
            temp_list= []
            for j in range(num_HybCycle*num_channels):
                if (j%num_HybCycle == 0) and (j != 0):  
                    reorg_dot_files.append(temp_list)
                    temp_list = []
                    temp_list.append(dot_path_list[j][i])
                else:
                    temp_list.append(dot_path_list[j][i])
                if j==(num_HybCycle*num_channels)-1:
                    reorg_dot_files.append(temp_list)

        #make textfile of new thresholds for later reference
        output_path_text = Path(reorg_dot_files[0][0]).parent.parent
        f = open(str(output_path_text) + f"/optimal_threshold_test_complete.txt" ,"w+")
        for element in reorg_dot_files:
            for i in range(len(element)):
                csv_name = str(element[i].name)
                csv_name = csv_name.split("_")
                threshold_extracted = csv_name[2][9:]
                ch_extracted = csv_name[3].replace(".csv","")[2:]
                f.write(threshold_extracted + "\t" + ch_extracted + "\n")
        f.close()

        #read files and concatenate
        df_list = []
        df_list_final = []
        for i in range(num_channels*11):
            if i%num_channels == 0 and i != 0:
                df_con_2=pd.concat(df_list)
                df_list_final.append(df_con_2)
                df_list=[]
                df_read = [pd.read_csv(str(j), index_col=0) for j in reorg_dot_files[i]]
                df_con = pd.concat(df_read)
                df_list.append(df_con)
            if i == ((num_channels*11)-1):
                df_read = [pd.read_csv(str(j), index_col=0) for j in reorg_dot_files[i]]
                df_con = pd.concat(df_read)
                df_list.append(df_con)
                df_con_2=pd.concat(df_list)
                df_list_final.append(df_con_2)
            else:
                df_read = [pd.read_csv(str(j), index_col=0) for j in reorg_dot_files[i]]
                df_con = pd.concat(df_read)
                df_list.append(df_con)

        #check if files are good
        for i in range(len(df_list_final)):
            if len(df_list_final[i]["hyb"].unique()) == num_HybCycle and len(df_list_final[i]["ch"].unique())== num_channels:
                continue
            else:
                print("dataframe missing information, dataframe number",i+1 )
                exit()
        print("Files passed initial QC for decoding")

        #write files
        for i in range(len(df_list_final)):
            comb = df_list_final[i]
            for z in range(num_z):
                comb_z = comb[comb["z"]==z]
                output_folder = dot_path_list[0][0].parent.parent.parent / "dots_comb" / "opt_thresh_folder"
                output_path = output_folder /f"MMStack_Pos{pos}"/f"Threshold_{i}"/"Dot_Locations"
                output_path.mkdir(parents=True, exist_ok=True)
                comb_z.to_csv(str(output_path) + f"/locations_z_{z}.csv", index=False)
        print("Files are ready for decoding")
    else:
        ch_df = []
        #loop through channels
        for i in range(num_channels):
            position_dots = f'MMStack_Pos{pos}_{i}.csv'
            #get all paths for specific position across hybs
            files_dots, _, _ = find_matching_files(path_dots, 'HybCycle_{hyb}' + f'/{position_dots}')
            files_dots = [str(f) for f in files_dots]
        
            dots = [pd.read_csv(j, index_col=0) for j in files_dots]
            con = pd.concat(dots)
            ch_df.append(con)
        #combine all channels    
        con_final = pd.concat(ch_df)
        
        #check if files are good
        if len(con_final["hyb"].unique()) == num_HybCycle and len(con_final["ch"].unique())== num_channels:
            print("Files passed initial QC for decoding")
        else:
            print("dataframe missing information:","HybCycle=",sorted(con_final["hyb"].unique()),
                  ",Num Channels=",sorted(con_final["ch"].unique()))
            exit()
        
        for z in range(num_z):
            con_z = con_final[con_final["z"]==z]
            output_folder = Path(path_dots).parent.parent / "dots_comb" /"final"
            output_path = output_folder /f"MMStack_Pos{pos}"/"Dot_Locations"
            output_path.mkdir(parents=True, exist_ok=True)
            con_z.to_csv(str(output_path) +f"/locations_z_{z}.csv", index=False)
        print("Data ready for decoding")     