"""
authors: Katsuya Lex Colon and Arun Chakravorty
group: Cai Lab
date:11/15/22
"""

#general packages
import plotly.express as px
from pathlib import Path
import numpy as np
import os
import skimage.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from util import pil_imread
from cellpose import plot
from cellpose import core
from cellpose import models
mpl.rcParams['figure.dpi'] = 300



def plot_2d(img, zmax):

    fig = px.imshow(
        img,
        width=600,
        height=600,
        binary_string=True,
        binary_compression_level=4,
        binary_backend='pil',
        zmax = zmax)
    
    fig.show()
    
def read_images(files, max_project=True):
    
    #Do you wish to max project?
    max_project = max_project

    #Read in images
    imgs = []
    for i in tqdm(range(len(files))):
        img = pil_imread(files[i], swapaxes=True)
        if max_project == True:
            imgs.append(np.max(img,axis=0))
        else:
            imgs.append(img)
            
    return imgs

def isolate_image(imgs, pos=0, channel=1, have_multiple_z=False):
    
    #isolate image
    if have_multiple_z == True:
        #get all z for specific channel
        img = imgs[pos]
        img_seg = img[channel]
        img_dapi = img[-1]
        img_seg = img.reshape(img_seg.shape[0],1,img_seg.shape[1],img_seg.shape[2])
        img_dapi = img.reshape(img_dapi.shape[0],1,img_dapi.shape[1],img_dapi.shape[2])
        img = np.concatenate([img_seg,img_dapi],axis=1)
    else:
        img_seg = imgs[pos][channel].reshape(1, imgs[pos][channel].shape[0], imgs[pos][channel].shape[1])
        img_dapi = imgs[pos][-1].reshape(1, imgs[pos][channel].shape[0], imgs[pos][channel].shape[1])
        img = np.concatenate([img_seg,img_dapi], axis=0)
            
    return img
            
def plot_isolated_image(img, have_multiple_z = False, zmax=5000):
    if have_multiple_z == True:
        z = int(input("which z slice do you want to see?: "))
        if img.shape[1] == 2:
            # Cytoplasmic Channel
            plot_2d(img[z][0], zmax=zmax)
            # Nuclear Channel
            plot_2d(img[z][1], zmax=zmax)
        else:
            plot_2d(img[z][0],zmax=zmax)
    else:
        if len(img.shape) > 2:
            # Cytoplasmic Channel
            plot_2d(img[0],zmax=zmax)
            # Nuclear Channel
            plot_2d(img[1],zmax=zmax)
        else:
            plot_2d(img, zmax=zmax)
                      
                      
def cellpose_settings(num_gpus=1):
    if num_gpus != None:
        core.use_gpu(gpu_number=num_gpus, use_torch=True)
    model_choice = input("Do you want to segment Nucleus or Cytoplasm (Type Cytoplasm or Nucleus): ") 

    if model_choice == "Cytoplasm":
        model_type = "cyto2"
    else:
        model_type = "nuclei"

    model = models.Cellpose(gpu=True, model_type=model_type)
    
    return model
     
def cellpose_plots(img, masks, flows, have_multiple_z = True, num_channels = 1, channels=[0,0]):
    
    if num_channels > 1:
        channel = int(input("which channel do you want to look at (0 or 1): "))
        
    if have_multiple_z == False and num_channels == 1:
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
        plt.tight_layout()
        plt.show()
    elif have_multiple_z == False and num_channels > 1:
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img[channel], masks, flows[0], channels=channels)
        plt.tight_layout()
        plt.show()
    elif have_multiple_z == True and num_channels == 1:
        z = int(input("which z do you want to look at? (0,1,2,3...): "))
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img[z][0], masks[z], flows[0][z], channels=channels)
        plt.tight_layout()
        plt.show()
    else:
        z = int(input("which z do you want to look at? (0,1,2,3...): "))
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img[z][channel], masks[z], flows[0][z], channels=channels)
        plt.tight_layout()
        plt.show()
        
def generate_final_images(imgs, have_multiple_z=False, channel=0):
    
    imgs_final = []
    for i in tqdm(range(len(imgs))):
        img = isolate_image(imgs, pos = i, channel = channel, have_multiple_z = have_multiple_z)
        imgs_final.append(img)
                
    return imgs_final
            
def write_masks(masks, files, save_dir, repeat_mask_multi_z = 0):
    
    #save images
    print("Saving masks in the following directory: ",save_dir)
    
    if (not os.path.exists(save_dir)):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        

    for idx,mask in enumerate(masks):
        file_name=os.path.splitext(os.path.basename(files[idx]))[0]
        if len(mask.shape) > 2:
            for z in range(len(mask)):
                mask_z=mask[z]
                #Output name for masks
                mask_output_name=save_dir+file_name.replace(".ome",f"_z{z}.tif")
                #Save mask as 16-bit in case this has to be used for detecting than 255 objects
                mask_z=mask_z.astype(np.uint16)
                skimage.io.imsave(mask_output_name, mask_z, check_contrast=False)
        else:
            if repeat_mask_multi_z == 0:
                #Output name for masks
                mask_output_name=save_dir+file_name+".tif"
                #Save mask as 16-bit in case this has to be used for detecting than 255 objects
                mask=mask.astype(np.uint16)
                skimage.io.imsave(mask_output_name,mask, check_contrast=False)
            else:
                for z in range(repeat_mask_multi_z):
                    #Output name for masks
                    mask_output_name=save_dir+file_name.replace(".ome",f"_z{z}.tif")
                    #Save mask as 16-bit in case this has to be used for detecting than 255 objects
                    mask=mask.astype(np.uint16)
                    skimage.io.imsave(mask_output_name,mask, check_contrast=False)
    print("")
    print("~Files saved~ :D")
    
