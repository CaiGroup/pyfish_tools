"""
author: Katsuya Lex Colon
updated: 03/18/22
"""

from skimage import registration
from scipy import ndimage
import tifffile as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from helpers.util import pil_imread


def dapi_alignment_single(ref, moving, num_channels):
    """A function to obtain translational offsets using phase correlation. Image input should have the format z,c,x,y.
    Parameters
    ----------
    ref: Hyb 0 image path
    moving: image you are trying to align path
    
    Output
    -------
    image (c,z,x,y)
    """
    
    #create output path
    parent = Path(moving).parent
    while "pyfish_tools" not in os.listdir(parent):
        parent = parent.parent
    output_folder = parent / "pyfish_tools" / "output"/ 'dapi_aligned'
    hybcycle = Path(moving).parent.name
    output_path = output_folder / hybcycle / Path(moving).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        image_ref = pil_imread(ref, num_channels = None, swapaxes = True)
        if image_ref.shape[1] != num_channels:
            image_ref = pil_imread(ref, num_channels = None, swapaxes = False)
            if image_ref.shape[0] == image_ref.shape[1]:
                image_ref = check_axis(image_ref)
            if image_ref.shape[1] != num_channels:
                raise Exception("Error reading image file, will try to read it another way")
    except:
        image_ref = pil_imread(ref, num_channels = num_channels, swapaxes = True)
        if image_ref.shape[1] != num_channels:
            image_ref = pil_imread(ref, num_channels = num_channels, swapaxes = False)
            if image_ref.shape[0] == image_ref.shape[1]:
                image_ref = check_axis(image_ref)
                
    try:
        image_moving = pil_imread(moving, num_channels = None, swapaxes = True)
        if image_moving.shape[1] != num_channels:
            image_moving = pil_imread(moving, num_channels = None, swapaxes = False)
            if image_moving.shape[0] == image_moving.shape[1]:
                image_moving = check_axis(image_moving)
            if image_moving.shape[1] != num_channels:
                raise Exception("Error reading image file, will try to read it another way")
    except:
        image_moving = pil_imread(moving, num_channels = num_channels, swapaxes = True)
        if image_moving.shape[1] != num_channels:
            image_moving = pil_imread(moving, num_channels = num_channels, swapaxes = False)
            if image_moving.shape[0] == image_moving.shape[1]:
                image_moving = check_axis(image_moving)
    
    #get dapi channel for reference and moving assuming it is at the end
    dapi_ref = image_ref.shape[1]-1
    dapi_moving = image_moving.shape[1]-1
    
    #max project dapi channel
    max_proj_ref = np.max(np.swapaxes(image_ref,0,1)[dapi_ref], axis=0)
    max_proj_moving = np.max(np.swapaxes(image_moving,0,1)[dapi_moving], axis=0)
    
    #calculate shift on max projected dapi
    shift,error,phasediff = registration.phase_cross_correlation(
        max_proj_ref,max_proj_moving, upsample_factor=20)
    
    #apply shift across z's on all channels
    layer = []
    for z in range(image_moving.shape[0]):
        c_list = []
        for c in range(image_moving.shape[1]):
            img = ndimage.shift(image_moving[z][c],shift)
            c_list.append(img)
        layer.append(c_list)
    corr_stack = np.array(layer)
    del layer
    #write images
    tf.imwrite(str(output_path), corr_stack)
    #write shift
    pos = output_path.name.split("_")[1].replace(".ome.tif","_shift.txt")
    shift_output = output_path.parent/pos
    np.savetxt(str(shift_output),shift)
    del corr_stack

def dapi_alignment_parallel(image_ref, images_moving, num_channels):
    """Run dapi alignment on all positions
    Parameter
    ---------
    image_ref: path to Hyb0
    images_moving: path to moving images
    z: optimal z slice"""

    import time
    start = time.time()
    
    if type(images_moving) != list:
        with ThreadPoolExecutor(max_workers=20) as exe:
            exe.submit(dapi_alignment_single, image_ref, images_moving, num_channels)
    
    else:
        with ThreadPoolExecutor(max_workers=20) as exe:
            futures = {}
            for path in images_moving:
                fut = exe.submit(dapi_alignment_single, image_ref, path, num_channels)
                futures[fut] = path
        
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
   
