"""
author: Katsuya Lex Colon
date: 07/21/22
group: CaiLab
"""

print("Loading packages...")

#general packages
import pandas as pd
import numpy as np
import ast
import sys
import time as ti
#image reading
import tifffile as tf
#file management packages
from glob import glob 
from pathlib import Path
#progression
from tqdm import tqdm
#import custom function
from helpers.util import pil_imread
from helpers.pre_processing import *
from helpers.stain_intron_quant_main import *
from helpers.map_spots_to_masks import *

###collect inputs from user and check if paths are correct
    
#get mask paths
timecourse = input("Is this a time course dataset (y/n)? ")
if timecourse == "y":
    #ask user if the parent directory has time info
    timeinfo = input("Does the parent directory for mask and images have time info (y/n)? ")
    if timeinfo == "n":
        sys.exit("Change parent directory to have time info.")

if timecourse == "n":
    cyto_masks = []
    while len(cyto_masks) == 0:
        cyto_mask_path = input("path to cytoplasm mask directory (path or empty): ")
        if cyto_mask_path != "":
            cyto_masks = list(Path(cyto_mask_path).glob("*.tif"))
            if len(cyto_masks) == 0:
                print("")
                print("Check file path, no files were found.")
                continue
            cyto_masks = sorted([str(file) for file in cyto_masks])
        else:
            break

else:
    cyto_masks = []
    while len(cyto_masks) == 0:
        cyto_mask_path = input("path to cytoplasm mask directory (path or empty): ")
        if cyto_mask_path != "":
            cyto_masks = glob(str(Path(cyto_mask_path) / "*"/ "*.tif"))
            if len(cyto_masks) == 0:
                print("")
                print("Check file path, no files were found.")
                continue
            cyto_masks = sorted([str(file) for file in cyto_masks])
        else:
            break
        
if timecourse == "n": 
    nuc_masks = []
    while len(nuc_masks) == 0:
        nuc_mask_path = input("path to nuclear mask directory (path or empty): ")
        if nuc_mask_path != "":
            nuc_masks = list(Path(nuc_mask_path).glob("*.tif"))
            if len(nuc_masks) == 0:
                print("")
                print("Check file path, no files were found.")
                continue
            nuc_masks = sorted([str(file) for file in nuc_masks])
        else:
            break
else:
    nuc_masks = []
    while len(nuc_masks) == 0:
        nuc_mask_path = input("path to nuclear mask directory (path or empty): ")
        if nuc_mask_path != "":
            nuc_masks = glob(str(Path(nuc_mask_path) / "*"/ "*.tif"))
            if len(nuc_masks) == 0:
                print("")
                print("Check file path, no files were found.")
                continue
            nuc_masks = sorted([str(file) for file in nuc_masks])
        else:
            break
#get image paths
if timecourse == "n":
    path_to_images = []
    while len(path_to_images) == 0:
        path_to_images = Path(input("path to image directory you want to quantify: "))
        path_to_images = list(path_to_images.glob("*.tif"))
        if len(path_to_images) == 0:
            print("")
            print("Check file path, no files were found.")
            continue
        path_to_images = sorted([str(file) for file in path_to_images])
else:
    path_to_images = []
    while len(path_to_images) == 0:
        path_to_images = Path(input("path to image directory you want to quantify: "))
        path_to_images = glob(str(Path(path_to_images) / "*"/ "*.tif"))
        if len(path_to_images) == 0:
            print("")
            print("Check file path, no files were found.")
            continue
        path_to_images = sorted([str(file) for file in path_to_images])

dual_channel_stain = input("Do you want to analyze 1 or 2 stains, excluding intron channel (1 or 2)? ")

if dual_channel_stain == "2":
    channel1 = input("Which channel first (1,2,3,4)? ")
    first_stain = str(input("What is the name of this stain? "))
    channel2 = input("Which channel second (1,2,3,4)? ")
    second_stain = str(input("What is the name of this stain? "))
else:
    channel1 = input("Which channel first (1,2,3,4)? ")
    first_stain = str(input("What is the name of this stain? "))
    channel2 = None
    
intron_detection = str(input("Do you want to quantify introns (y/n)? "))

if intron_detection == "y":
    intron_channel = int(input("Which channel has introns (1,2,3,4)? "))
    size_cutoff = str(input("Do you want to remove large blobs (y/n)? "))
    if size_cutoff == "y":
        size_cutoff = 3
    else:
        size_cutoff = None
    intron_threshold = float(input("What threshold for intron detection (scale=0.01-1; type 0.02 for detecting most spots except really dim ones)? "))

swapaxes = ast.literal_eval(input("Does c and z axes need to be swapped for raw image (True/False)? "))

#read, max project images, and check axes
img1 = pil_imread(path_to_images[0], swapaxes=swapaxes)
max_img1 = np.max(img1, axis=0)
shape_correct = input(f"Found {max_img1.shape[0]} channels. Is this correct (y/n)? ")

while shape_correct != "y":
    swapaxes = not swapaxes
    img1 = pil_imread(path_to_images[0], swapaxes = swapaxes)
    max_img1 = np.max(img1, axis=0)
    shape_correct = input(f"Found {max_img1.shape[0]} channels. Is this correct (y/n)? ")
    if shape_correct == "n":
        swapaxes = not swapaxes
        
fitting_line = ast.literal_eval(input("Do you want to force intercept to 0 (True/False)? "))

###All file paths should have similar naming convention so sorting should match
print("Will begin analysis...PRAY")
start = ti.time()
print("Reading masks...")
#read masks
if cyto_mask_path != "" and nuc_mask_path != "":
    cyto_mask_img = [tf.imread(mask) for mask in tqdm(cyto_masks)]
    nuc_mask_img = [tf.imread(mask) for mask in tqdm(nuc_masks)]
else:
    for mask_src in ["cyto_masks","nuc_masks"]:
        if mask_src in locals():
            main_src = mask_src
            mask = [tf.imread(str(k)) for k in tqdm(eval(mask_src))]
            
print("Reading and max projecting image...")
        
images = []
for img in tqdm(path_to_images):
    img1 = pil_imread(img, swapaxes = swapaxes)
    max_img1 = np.max(img1, axis=0)
    images.append(max_img1)
    
print("Quantifying stains...")

if "mask" in locals():
    #quantify masks
    df_list = []
    for i in tqdm(range(len(mask))):
        pos = Path(path_to_images[i]).name
        pos_num = pos.find("Pos")
        pos = pos[pos_num+3:].replace(".ome.tif","")
        if channel2 != None:
            df = quant_mask(images[i], mask[i], pos, channel1=int(channel1), channel2=int(channel2))
        else:
            df = quant_mask(images[i], mask[i], pos, channel1=int(channel1), channel2=channel2)
        if len(df.columns) == 3:
            df.columns = ["Cell id", f"{first_stain}", f"{second_stain}"]
        else:
            df.columns = ["Cell id", f"{first_stain}"]
        if timecourse == "y":
            time = Path(path_to_images[i]).parent.name
            df["Time"] = time
        df_list.append(df)
        #write record of images being compared
        with open("./results/file_log.txt","a") as f:
            f.write(eval(main_src)[i] + "\n")
            f.write(path_to_images[i] + "\n")
            f.write("\n")
            f.close()
    final = pd.concat(df_list).reset_index(drop=True)
    if "second_stain" not in locals():
        second_stain = "none"
    output_file = Path(f"./results/{first_stain}_vs_{second_stain}.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(str(output_file))
    if timecourse == "y" and second_stain != "none":
        gen_figure(final, time_course=True, no_intercept=fitting_line)
    elif timecourse == "n" and second_stain != "none":
        gen_figure(final, time_course=False, no_intercept=fitting_line)
        
else:
    #first to cyto mask quantification
    df_list_cyto = []
    for i in tqdm(range(len(cyto_mask_img))):
        pos = Path(path_to_images[i]).name
        pos_num = pos.find("Pos")
        pos = pos[pos_num+3:].replace(".ome.tif","")
        if channel2 != None:
            df = quant_mask(images[i], cyto_mask_img[i], pos, channel1=int(channel1), channel2=int(channel2))
        else:
            df = quant_mask(images[i], cyto_mask_img[i], pos, channel1=int(channel1), channel2=channel2) 
        if len(df.columns) == 3:
            df.columns = ["Cell id", f"{first_stain}", f"{second_stain}"]
        else:
            df.columns = ["Cell id", f"{first_stain}"]
        if timecourse == "y":
            time = Path(path_to_images[i]).parent.name
            df["Time"] = time
        df_list_cyto.append(df)
        #write record of images being compared
        with open("./results/file_log.txt","a") as f:
            f.write(cyto_masks[i] + "\n")
            f.write(path_to_images[i] + "\n")
            f.write("\n")
            f.close()
    final_cyto = pd.concat(df_list_cyto).reset_index(drop=True)
    if "second_stain" not in locals():
        second_stain = "none"
    output_file = Path(f"./results/cyto_{first_stain}_vs_{second_stain}.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_cyto.to_csv(str(output_file))
    if timecourse == "y":
        gen_figure(final_cyto, time_course=True, no_intercept=fitting_line)
    else:
        gen_figure(final_cyto, time_course=False, no_intercept=fitting_line)
    
    #now do nuclear mask quantification
    df_list_nuc = []
    for i in tqdm(range(len(nuc_mask_img))):
        pos = Path(path_to_images[i]).name
        pos_num = pos.find("Pos")
        pos = pos[pos_num+3:].replace(".ome.tif","")
        if channel2 != None:
            df = quant_mask(images[i], nuc_mask_img[i], pos, channel1=int(channel1), channel2=int(channel2))
        else:
            df = quant_mask(images[i], nuc_mask_img[i], pos, channel1=int(channel1), channel2=channel2) 
        if len(df.columns) == 3:
            df.columns = ["Cell id", f"{first_stain}", f"{second_stain}"]
        else:
            df.columns = ["Cell id", f"{first_stain}"]
        if timecourse == "y":
            time = Path(path_to_images[i]).parent.name
            df["Time"] = time
        df_list_nuc.append(df)
        #write record of images being compared
        with open("./results/file_log.txt","a") as f:
            f.write(nuc_masks[i] + "\n")
            f.write(path_to_images[i] + "\n")
            f.write("\n")
            f.close()
    final_nuc = pd.concat(df_list_nuc).reset_index(drop=True)
    if "second_stain" not in locals():
        second_stain = "none"
    output_file = Path(f"./results/nuc_{first_stain}_vs_{second_stain}.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_nuc.to_csv(str(output_file))
    if timecourse == "y":
        gen_figure(final_nuc, time_course=True, no_intercept=fitting_line)
    else:
        gen_figure(final_nuc, time_course=False, no_intercept=fitting_line)
     
    #remove cyto masks since we won't need it anymore
    del cyto_mask_img
                   
    #reassign variable name for intron detection if applicable
    mask = nuc_mask_img

#perform intron dot detection
if intron_detection == "y":
    print("Will begin intron detection...")
    with open("./results/file_log.txt", "w+") as f:
        f.close()
    dots = []
    for i in tqdm(range(len(images))):
        #perform high pass gaussian filter, low pass gaussian filter, then image scaling 
        img_corr = bkgrd_corr_one(images[i], stack_bkgrd=None, correction_type = Gaussian_and_Gamma_Correction,
                   gamma = 1.0, kern_hpgb=7, sigma = 15, rb_radius=5, p_min=80,
                   p_max = 99.999, norm_int = True, rollingball = False, 
                   lowpass=True, match_hist=False, subtract=True, divide=False, tophat_raw=False)
        #write preprocessed image for reference
        output_path = Path("./results/pre_processed_images") / Path(path_to_images[i]).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tf.imwrite(str(output_path), img_corr)
        #dot detect on corrected image
        spots = dot_detection(img_corr, size_cutoff = size_cutoff, 
                      threshold = intron_threshold , channel = intron_channel)
        #write record of images being compared
        with open("./results/file_log.txt","a") as f:
            f.write(eval(main_src)[i] + "\n")
            f.write(path_to_images[i] + "\n")
            f.write("\n")
            f.close()
        #collect position number info
        pos_info = Path(path_to_images[i]).name
        pos_num = pos_info.find("Pos")
        pos_info = pos_info[pos_num+3:].replace(".ome.tif","")
        spots = keep_dots_in_cells(mask[i], spots, pos_info=pos_info)
        #collect time info if applicable
        if timecourse == "y":
            time_info = Path(path_to_images[i]).parent.name
            spots["Time"] = time_info
        dots.append(spots)
    
    #combine all spots across time (if apllicable) and positions
    final_dots = pd.concat(dots).reset_index(drop=True)
    #get intron counts per time if applicable
    if timecourse == "y":
        intron_counts_per_time = pd.DataFrame(final_dots.groupby("Time").size())
        intron_counts_per_time.columns = ["Total Intron Counts"]
        output_file = Path("./results/intron_spot_counts_by_time.csv")
        intron_counts_per_time.to_csv(str(output_file))
    #write file
    output_file = Path("./results/intron_spot_info.csv")
    final_dots.to_csv(str(output_file))

print("Analysis finished...")
print(f"This task took {round((ti.time()-start)/60,2)} min")
