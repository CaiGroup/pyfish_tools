"""
author: Katsuya Lex Colon
group: Cai Lab
date: 02/03/22
"""

#general package
import numpy as np
import time
#image processing
from skimage.segmentation import clear_border, find_boundaries
import tifffile as tf
#file management
from pathlib import Path
#parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

def edge_deletion(img_src, output_dir = None):
    """
    This function will delete 2 pixels between masks that touch, and 
    remove masks that are on the borders
    
    Parameters
    ----------
    img_src = location of labeled image
    output_dir = string of output directory
    """
    #read image
    labeled_img = tf.imread(img_src)
    #remove masks touching edge of image
    remove_border = clear_border(labeled_img)
    #find borders of masks while identifying masks that touch
    outline = find_boundaries(remove_border, mode='outer')
    #invert boolean mask and delete masks that touch
    new_masks = np.invert(outline) * remove_border
    
    #make output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    #take image name
    img_name = Path(img_src).name
    
    #final path
    output_path = output_dir / img_name
    
    #write mask
    tf.imwrite(str(output_path), new_masks)
    
def edge_deletion_parallel(img_src_list, output_dir=None):
    """
    Will perform edge deletion in parallel provided a list of image paths
    Parameters
    ----------
    img_src_list = locations of labeled images
    output_dir = string of output directory
    """
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=32) as exe:
        futures = {}
        for img in img_src_list:
            fut = exe.submit(edge_deletion, img, output_dir = output_dir)
            futures[fut] = img

        for fut in as_completed(futures):
            img = futures[fut]
            print(f'{img} completed after {(time.time() - start)/60} seconds')