"""
author: Katsuya Lex Colon
group: Cai Lab
date: 06/10/22
"""

#general package
import numpy as np
import time
#image processing
from skimage.segmentation import clear_border, find_boundaries
from scipy import ndimage
import tifffile as tf
#file management
import os
from pathlib import Path
#parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

def edge_deletion(img_src, output_dir = None, have_seg_img = True):
    """
    This function will delete 2 pixels between masks that touch, and 
    remove masks that are on the borders
    
    Parameters
    ----------
    img_src = location of labeled image
    output_dir = string of output directory
    have_seg_img = bool for whether you have segmentation image
    
    """
    #read image
    labeled_img = tf.imread(img_src)
    #remove masks touching edge of image
    new_masks = clear_border(labeled_img)
    
    if have_seg_img == True:
        #find shift path 
        src = Path(img_src)
        parent = src.parent
        while "dapi_aligned" not in os.listdir(parent):
            parent = parent.parent
        #get segmentation shift 
        shift_path = parent/"dapi_aligned"/"segmentation"
        #check if file is present
        assert os.path.isdir(str(shift_path)), "Make sure you have a folder called segmentation in dapi_aligned directory"
        #get pos info
        pos = src.name.split("_")[1].split(".")[0]
        pos_name = pos + "_shift.txt"
        #read in shift
        shift = np.loadtxt(str(shift_path/pos_name))
        #apply shift to mask
        new_masks_2 = ndimage.shift(new_masks,shift.astype(int), order=3)
        #remove shift artifacts
        artifacts = set(np.unique(new_masks_2)) - set(np.unique(new_masks))
        for val in artifacts:
            new_masks_2[new_masks_2==val]=0
        #delete edges again if shift caused a mask to be on the edge
        new_masks = clear_border(new_masks_2)
        
    #find borders of masks while identifying masks that touch
    outline = find_boundaries(new_masks, mode='outer')
    #invert boolean mask and delete masks that touch
    final_masks = np.invert(outline) * new_masks
    
    #make output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #take image name
    img_name = Path(img_src).name
    
    #final path
    output_path = output_dir / img_name
    
    #write mask
    tf.imwrite(str(output_path), final_masks)
    
def edge_deletion_parallel(img_src_list, output_dir=None,  have_seg_img = True):
    """
    Will perform edge deletion in parallel provided a list of image paths
    Parameters
    ----------
    img_src_list = locations of labeled images
    output_dir = string of output directory
    have_seg_img = bool for whether you have segmentation image
    """
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=32) as exe:
        futures = {}
        for img in img_src_list:
            fut = exe.submit(edge_deletion, img, output_dir = output_dir,  have_seg_img = have_seg_img)
            futures[fut] = img

        for fut in as_completed(futures):
            img = futures[fut]
            print(f'{img} completed after {(time.time() - start)/60} seconds')
