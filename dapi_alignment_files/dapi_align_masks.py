"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 12/03/21
"""

from skimage import registration
from scipy import ndimage
import tifffile as tf
import plotly.express as px
import numpy as np
from tqdm import tqdm
from pathlib import Path

def plot_dapi(img, zmax):
    """Function to generate plots with slide panel
    Parameters:
    -----------
    img = image containing ref and corrected
    zmax= set maximum intensity"""
    
    #For Plotting 2d image
    #-------------------------------------------
    fig = px.imshow(
        img,
        width=700,
        height=700,
        binary_string=True,
        binary_compression_level=4,
        animation_frame=0,
        binary_backend='pil',
        zmax = zmax)
    
    fig.show()

def dapi_alignment(ref, moving, output_dir, cyto_channel=0, zmax=3000, all_channels=False):
    """A function to obtain translational offsets using phase correlation. Image input should have the format z,c,x,y.
    Parameters
    ----------
    ref: Hyb 0 image
    moving: image you are trying to align
    output_dir: path for outputing images. Will not output if single image.
    zmax = max pixel int. Only works if one image is analyzed.
    
    Output
    -------
    image (c,z,x,y)
    """
    #check to see if it is list of images
    if type(moving) == list:
        if all_channels==False:
            shift_list = []
            #get dapi channel for reference and moving assuming it is at the end
            dapi_ref = ref[0].shape[1]-1
            dapi_moving = moving[0].shape[1]-1

            #calculate shift
            print("Getting shifts...")
            for i in range(len(moving)):
                #max project dapi channel 
                max_proj_ref = np.max(np.swapaxes(ref[i],0,1)[dapi_ref], axis=0)
                max_proj_moving = np.max(np.swapaxes(moving[i],0,1)[dapi_moving], axis=0)
            
                shift,error,phasediff = registration.phase_cross_correlation(
                    max_proj_ref,max_proj_moving, upsample_factor=20)
                shift_list.append(shift)
            #apply shift across z's on dapi and cyto channel
            print("Applying shifts...")
            for i in tqdm(range(len(shift_list))):
                layer = []
                for j in range(moving[0].shape[0]):
                    img_cyto = ndimage.shift(moving[i][j][cyto_channel],shift_list[i])
                    img_nuc = ndimage.shift(moving[i][j][dapi_moving],shift_list[i])
                    layer.append(np.array([img_cyto,img_nuc]))
                new_image_stack = np.array(layer)
                #write images
                tf.imwrite(output_dir+"/MMStack_Pos{}.ome.tif".format(i),new_image_stack)
        else:
            shift_list = []
            #get dapi channel for reference and moving assuming it is at the end
            dapi_ref = ref[0].shape[1]-1
            dapi_moving = moving[0].shape[1]-1
            
            #calculate shift
            print("Getting shifts...")
            for i in range(len(moving)):
                #max project dapi channel 
                max_proj_ref = np.max(np.swapaxes(ref[i],0,1)[dapi_ref], axis=0)
                max_proj_moving = np.max(np.swapaxes(moving[i],0,1)[dapi_moving], axis=0)
                
                shift,error,phasediff = registration.phase_cross_correlation(
                    max_proj_ref, max_proj_moving, upsample_factor=20)
                shift_list.append(shift)
                
            #apply shift across z's on all channels
            print("Applying shifts...")
            for i in range(len(shift_list)):
                layer = []
                for j in range(moving[0].shape[0]):
                    c_list = []
                    for c in range(moving[0].shape[1]):
                        img = ndimage.shift(moving[i][j][c],shift_list[i])
                        c_list.append(img)
                    layer.append(c_list)
                corr_stack = np.array(layer)
                #write images
                tf.imwrite(output_dir+"/MMStack_Pos{}.ome.tif".format(i),corr_stack)
    else:
        #get dapi channel for reference and moving assuming it is at the end
        dapi_ref = ref.shape[1]-1
        dapi_moving = moving.shape[1]-1
        
        #max project dapi channel
        max_proj_ref = np.max(np.swapaxes(ref,0,1)[dapi_ref], axis=0)
        max_proj_moving = np.max(np.swapaxes(moving,0,1)[dapi_moving], axis=0)
            
        #calculate shift
        shift,error,phasediff = registration.phase_cross_correlation(
            max_proj_ref,max_proj_moving, upsample_factor=20)
        #apply shift 
        img_nuc = ndimage.shift(moving[z][dapi_moving],shift)  
        
        plot_dapi(np.array([ref[z][dapi_ref],img_nuc]),zmax)
        
