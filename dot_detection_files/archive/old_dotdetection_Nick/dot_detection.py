import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datapipeline.dot_detection.dot_detectors_3d import hist_jump_3d

#Run Dot Detection - returns df of dots and tiff file 
#-----------------------------------
def dot_detection(tiff_src, num_channels=4):
    """Uses general dot detection code with blob log and performs intensity cutoffs by performing biggest jump analysis
    
    Parameters
    ----------
    tiff_src = image path for dot detection
    
    Returns
    -------
    csv file of dot locations by position and at which hyb
    """
    #generate output folder and apth
    orig_image_dir = Path(tiff_src).parent.parent
    output_folder = Path(orig_image_dir).with_name('dots_detected')
    output_path = output_folder / Path(tiff_src).relative_to(orig_image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_path_csv = str(output_path).replace(".ome.tif",".csv")
    
    #run dot detection on each channel
    list_df = []
        
    for i in range(num_channels):
        df_tiff = hist_jump_3d.get_dots_for_tiff(tiff_src, 
                                               channels_to_detect_dots = [i+1], 
                                               
                                               #Params for Biggest Jump
                                               #----------------------------
                                               strictness = 10,
                                               nbins = 100,
                                               #----------------------------
                                               
                                               #Parameters that go into blob log 
                                               #----------------------------
                                               min_sigma = 1.5,
                                               max_sigma = 5, 
                                               num_sigma = 5,
                                               threshold = 0.001,
                                               overlap = .6,
                                               #----------------------------
                                               
                                               #Background Subtraction
                                               #----------------------------
                                               bool_background_subtraction = False,
                                               
                                               #Remove Super Brights, Setting to "False" does not remove any brights
                                               #----------------------------
                                               bool_remove_bright_dots = False,
                                               #----------------------------
                                               
                                               
                                               #Preprocess params
                                               #----------------------------
                                               bool_blur = False,
                                               blur_kernel_size = 3,
                                               
                                               bool_rolling_ball = False,
                                               rolling_ball_kernel_size = 20,
                                               
                                               bool_tophat = False,
                                               tophat_kernel_size = 4
                                               #----------------------------                                               
                                              )
        list_df.append(df_tiff[0])
        
    df_final = pd.concat(list_df, ignore_index=True)
    #output as csv
    df_final.to_csv(output_path_csv)
                                  
def dot_detection_parallel(images, num_channels=4):
    """Run dot detection on all positions
    Parameter
    ---------
    images: path to images
    num_channels: number of channels in image
    """

    import time
    start = time.time()
    if type(images) != list:
        with ProcessPoolExecutor(max_workers=12) as exe:
            exe.submit(dot_detection, images, num_channels)
    
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(dot_detection, path, num_channels)
                futures[fut] = path
        
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')