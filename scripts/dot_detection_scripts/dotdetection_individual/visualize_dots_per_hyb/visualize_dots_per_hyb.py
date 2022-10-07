"""
author: Katsuya Lex Colon
lab: Cai Lab
date: 04/08/22
"""

#general packages
import pandas as pd
import tifffile as tf
import numpy as np
#image correction
from skimage.exposure import equalize_adapthist
#plotting package
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def plot_2d_locs_on_2d_image(df_locs_2d_1, img_2d, zmax=1000):
    
    #For Plotting 2d image
    #-------------------------------------------
    fig = px.imshow(
        img_2d,
        width=700,
        height=700,
        binary_string=True,
        binary_compression_level=4,
        binary_backend='pil',
        zmax = zmax
    )
    #-------------------------------------------
    
    #For Plotting 2d dots
    #-------------------------------------------
    fig.add_trace(go.Scattergl(
        x=df_locs_2d_1.x,
        y=df_locs_2d_1.y,
        mode='markers',
        marker_symbol='cross',
        marker=dict(
            #maxdisplayed=1000,
            size=4
            ),
        name = "Dots"
        )
    )
    
    fig.show()
    
def plot_3d_locs_on_2d_image(df_tiff_1, channel, raw_src = None, zmax=10, z_slice_range = (0,4)):
    
    #read image
    tiff = tf.imread(raw_src)
    if len(tiff.shape) == 3:
        tiff = tiff.reshape(1,tiff.shape[0],tiff.shape[1],tiff.shape[2])
    correct_shape = input(f"This image has {tiff.shape[1]} channels, is this correct (y/n)?")
    if correct_shape != "y":
        tiff = np.swapaxes(tiff, 0,1)
    #get specific z slice
    tiff = tiff[z_slice_range[0]:z_slice_range[1],:,:,:]
    print(f"Your new shape is: {tiff.shape}")
    #plot
    for z in range(len(tiff[:,channel])):
        df_locs_2d_1 = df_tiff_1[(df_tiff_1.z > z-1) & (df_tiff_1.z < z+1)]
        plot_2d_locs_on_2d_image(df_locs_2d_1, tiff[z, channel], zmax=zmax)
            
def crop_region_and_coord(img_src, locations, hyb=0, z=0,ch=1,xrange=(650,850),yrange=(850,1050)):
    
    """
    Crop image slice and get adjusted coordinates
    
    Parameters
    -----------
    img_src = path to image
    locations = path to dot locations file matching image hyb
    z = which z slice
    ch = which channel
    xrange = tuple xmin and xmax
    yrange = tuple ymin and ymax
    """
    #read image
    img = tf.imread(img_src)
    
    #read locations file
    locations = pd.read_csv(locations)
    #slice out desired info
    locations = locations[(locations["hyb"]==(hyb)) &
                          (locations["z"]==(z)) & 
                          (locations["ch"]==(ch))]
    
    #slice out region
    img_slice = img[z,ch-1,yrange[0]:yrange[1],xrange[0]:xrange[1]]
    
    #adjust contrast
    img_slice = equalize_adapthist(img_slice)
    
    #get coordinates and adjust to img slice
    loc_upper = locations[(locations["x"]<xrange[1]) & (locations["y"]<yrange[1])]
    loc_clipped = loc_upper[(loc_upper["x"]>xrange[0]) & (loc_upper["y"]>yrange[0])]
    new_x = loc_clipped["x"]-xrange[0]
    new_y = loc_clipped["y"]-yrange[0]
    
    return img_slice, new_x, new_y

def plot_dots_all_hybs(img_src_list, locations_file, 
                       z=0,ch=1,xrange=(650,850),yrange=(850,1050), 
                       num_hybs=24, vmax=0.5, filename = None):
    
    """
    Plot dots over all hybs for a given region
    
    Parameters
    ----------
    img_src_list = list of image paths
    locations_list = list of dot locations file matching image hyb
    z = which z slice
    ch = which channel
    xrange = tuple xmin and xmax
    yrange = tuple ymin and ymax
    num_hybs = number of hybs you are trying to visualize
    vmax = 0-1 for contrast adjustment
    filename = name of png if export is desired
    """
    
    #if the number of hybs is greater than 3 then make square-like subplot
    if num_hybs > 3:
        x = round(np.sqrt(num_hybs))
        #generate subplot canvas
        fig, axs = plt.subplots(x, x, figsize=(20,20), sharex = True, sharey = True)
    #if not, then make 1 row
    else:
        x=1
        y=num_hybs
        fig, axs = plt.subplots(x, y, figsize=(20,20), sharex = True, sharey = True)
    
    #go through each ax and plot dots on img slice for slect hybs
    for i, ax in enumerate(fig.axes):
        if i != num_hybs:
            try:
                #get slice, along with x and y coordinates
                img_slice, new_x, new_y = crop_region_and_coord(img_src_list[i], locations_file, 
                                                                hyb = i, z = z,ch = ch,
                                                                xrange = xrange,yrange = yrange)
                #plot image
                ax.imshow(img_slice, cmap="gray", vmax=vmax, zorder=1)
                #plot dots
                ax.scatter(new_x,new_y, s=2,zorder=2)
                ax.set_xlim(0,img_slice.shape[0])
                ax.set_ylim(0,img_slice.shape[0])
                ax.title.set_text(f'HybCycle_{i}')
            except IndexError:
                #if out of bounds then delete empty subplots
                ax.remove()      
        else:
            ax.remove()
    
    if filename != None:
        plt.draw()
        fig.savefig(filename, dpi=200)
        
    plt.show()
    plt.clf()