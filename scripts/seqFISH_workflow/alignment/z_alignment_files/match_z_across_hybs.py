"""
author: Katsuya Lex Colon
date: 09/07/22
"""

#data processing packages
import numpy as np
import re
import cv2
import tifffile as tf
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from helpers.util import pil_imread
#file management
from glob import glob
import os
from pathlib import Path

def z_matching(image_dir, pos_number = 0):
    
    """
    Function to find matching z across hybs for a given pos.
    
    Parameters
    ----------
    image_dir: directory of where your images are located
    pos_number: the position you wish to align
    """
    
    #output directory
    parent = Path(image_dir).parent
    while "pyfish_tools" not in os.listdir(parent):
        parent = parent.parent
    
    output_dir = parent / "pyfish_tools" / "output"/ "z_matched_images"
    
    #adjust string path if missing "/"
    if image_dir[-1] != "/":
        image_dir = image_dir + "/"
        
    #get all hybcycles
    hyb_images = glob(image_dir + f"HybCycle_*/MMStack_Pos{pos_number}.ome.tif")
    #organize hybcycles numerically
    key = [int(re.search('HybCycle_(\\d+)', f).group(1)) for f in hyb_images]
    files = list(np.array(hyb_images)[np.argsort(key)])
    
    #get ref image (which is hyb0)
    ref_path = files[0]
    
    print(f"reference image is: {ref_path}")
    
    #remove first line in files list
    del files[0]
    
    #collect matching z info by performing normalized correlation analysis
    match_z = []
    ref = pil_imread(ref_path, swapaxes=False)
    for file in files:
        hyb_list = []
        src = pil_imread(file, swapaxes=False)
        dapi_ch = ref.shape[1]-1
        for z in range(ref.shape[0]):
            #collect correlation info
            ccoef_list = []
            for z_2 in range(ref.shape[0]):
                ref_compressed = ref[z][dapi_ch].astype(np.float32)
                src_compressed = src[z_2][dapi_ch].astype(np.float32)
                corr = cv2.matchTemplate(ref_compressed, src_compressed, cv2.TM_CCOEFF_NORMED)
                ccoef_list.append(corr)
            #find best z
            best_z = np.argmax(ccoef_list)
            hyb_list.append([z,best_z])
        match_z.append(hyb_list)
        
    #check what the maximum allowed z stack can be
    unique_value_across_hybs = []
    i=0
    num_z = ref.shape[0]
    for _ in range(len(files)):
        hyb_slice = np.vstack(match_z)[:,1][i:i+num_z]
        i += num_z
        unique_value_across_hybs.append(len(np.unique(hyb_slice)))
    max_z = min(unique_value_across_hybs)
    
    #check where the reference image should begin
    check_z = []
    for z in match_z:
        flatten = [z[i][1] for i in range(len(z))]
        check = 0 
        while "first_z" not in locals():
            try:
                first_z = len(flatten)-1-flatten[::-1].index(check)
            except:
                check += 1
        check_z.append(first_z)
        del first_z
    ref_start = max(check_z)
    
    #now offset the zs in ref
    ref = pil_imread(ref_path, swapaxes=False)
    hyb_folder = Path(ref_path).parent.name
    output_path = output_dir / hyb_folder
    output_path.mkdir(parents=True, exist_ok=True)
    ref = ref[ref_start:max_z,:,:,:]
    if len(ref) != 0:
        tf.imwrite(str(output_path / f"MMStack_Pos{pos_number}.ome.tif" ), ref)
    print(len(np.argwhere(np.array(check_z) == max_z)))
    #write empty file
    output_info_path = output_dir / "matched_z_info"
    output_info_path.mkdir(parents=True, exist_ok=True)
    with open(str(output_info_path/f"pos{pos_number}_matched_z_info.txt"), "w+") as f:
        if len(ref) == 0:
            for bad_hyb in np.argwhere(np.array(check_z) == ref_start):
                f.write(f"Messed up z in Hyb{bad_hyb[0]+1}.\n")
            f.close()
            return 
        else:
            f.close()
    
    #now offset other hybs
    for i in range(len(match_z)):
        src = pil_imread(files[i], swapaxes=False)
        hyb_folder = Path(files[i]).parent.name
        flatten = [z[1] for z in match_z[i]]
        #only get matching z for sliced reference
        z_slice = flatten[ref_start:max_z]
        #write matched z info
        k = 0
        with open(str(output_info_path/f"pos{pos_number}_matched_z_info.txt"), "a") as f:
            for ref_z in np.arange(ref_start, max_z,1):
                f.write(f"For Hyb {i+1}, ref z_{ref_z} == moving z_{z_slice[k]}\n")
                k += 1
            f.close()
        offset_image = []
        for j in range(len(z_slice)):
            offset_image.append(src[z_slice[j],:,:,:])
        offset_image = np.array(offset_image)     
        output_path = output_dir / hyb_folder
        output_path.mkdir(parents=True, exist_ok=True)
        tf.imwrite(str(output_path / f"MMStack_Pos{pos_number}.ome.tif" ), offset_image)