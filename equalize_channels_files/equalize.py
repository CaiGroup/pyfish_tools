import numpy as np
import pandas as pd
import tifffile as tf
from skimage import exposure
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def equalize_channels(img, across_ch=True):
    """A function to normalize intensities within channel followed by across channels. 
    Across channel norm can be ignored if across_ch is set to false.
    Parameters
    ----------
    img = z,c,x,y
    across = bool
    """
    #normalize intensity within channel
    eq_img = []
    for z in range(img.shape[0]):
        z_slice = []
        for c in range(img.shape[1]):
            apt_hist = exposure.equalize_adapthist(img[z][c])
            z_slice.append(apt_hist)
        eq_img.append(z_slice)
    eq_img=np.array(eq_img)
    
    if across_ch == True:
        #normalize across channels
        across_norm = []
        for z in range(eq_img.shape[0]):
            z_slice = []
            for c in np.arange(0,eq_img.shape[1],1):
                if c != 0: 
                    match = exposure.match_histograms(eq_img[z][c],eq_img[z][0])
                    z_slice.append(match)
                else: 
                    z_slice.append(eq_img[z][c])
            across_norm.append(z_slice)

        across_norm = np.array(across_norm)
        return across_norm
    else:
        #just return normalized within channel
        return eq_img

def equalize_across(ref,img, across_ch=True):
    """A function to normalize intensities to a reference image"""
    
    orig_image_dir = Path(img).parent
    output_folder = Path(orig_image_dir).parent.parent / 'equalized'
    output_path = output_folder / Path(img).relative_to(orig_image_dir.parent)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ref = np.swapaxes(tf.imread(ref),0,1)
    img = np.swapaxes(tf.imread(img),0,1)
    #equalize across channels first
    ref_eq = equalize_channels(ref, across_ch = across_ch)
    img_eq = equalize_channels(img, across_ch = across_ch)
    
    #normalize to position 0
    across_pos = []
    for z in range(ref_eq.shape[0]):
        z_slice = []
        for c in range(ref_eq.shape[1]): 
            match = exposure.match_histograms(img_eq[z][c],ref_eq[z][c])
            z_slice.append(match)
        across_pos.append(z_slice)
        
    across_pos = np.array(across_pos)
    
    across_pos = exposure.rescale_intensity(across_pos, out_range=np.int16)

    tf.imwrite(output_path, across_pos)

def equalize_parallel(ref, img, across_ch=True):
    """Parallelize channel equalization"""
    
    import time
    start = time.time()
    
    if type(img) != list:
        with ProcessPoolExecutor(max_workers=12) as exe:
            exe.submit(equalize_across, ref, img, across_ch)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in img:
                fut = exe.submit(equalize_across,ref,path, across_ch)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')