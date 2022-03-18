"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 03/18/22
"""

from skimage import registration
from scipy import ndimage
import tifffile as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from util import pil_imread


def dapi_alignment_single(ref,moving):
    """A function to obtain translational offsets using phase correlation. Image input should have the format z,c,x,y.
    Parameters
    ----------
    ref: Hyb 0 image path
    moving: image you are trying to align path
    
    Output
    -------
    image (c,z,x,y)
    """
    orig_image_dir = Path(moving).parent.parent
    output_folder = Path(orig_image_dir) / 'notebook_pyfiles' / 'dapi_aligned'
    output_path = output_folder / Path(moving).parent.name / Path(moving).name
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image_ref = pil_imread(ref, swapaxes=True)
    image_moving = pil_imread(moving, swapaxes=True)
    
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
    tf.imwrite(str(output_path),np.swapaxes(corr_stack,0,1))
    #write shift
    pos = output_path.name.split("_")[1].replace(".ome.tif","_shift.txt")
    shift_output = output_path.parent/pos
    np.savetxt(str(shift_output),shift)
    del corr_stack

def dapi_alignment_parallel(image_ref,images_moving):
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
            exe.submit(dapi_alignment_single, image_ref, images_moving)
    
    else:
        with ThreadPoolExecutor(max_workers=20) as exe:
            futures = {}
            for path in images_moving:
                fut = exe.submit(dapi_alignment_single, image_ref, path)
                futures[fut] = path
        
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
   
