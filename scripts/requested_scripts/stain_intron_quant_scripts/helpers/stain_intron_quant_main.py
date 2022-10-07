#general packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from skimage.filters import threshold_local
from scipy.stats import norm
import sys
import os
#dot detection
from photutils.detection import DAOStarFinder
#plotting packages
import matplotlib.pyplot as plt
import seaborn as sns
#import custom function
from helpers.util import pil_imread
#ignore general warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


###custom functions

def daofinder(data,  threshold, fwhm = 4.0):
    """
    This function will return the output of daostarfinder
    Parameters
    ----------
    data = 2D array
    threshold = absolute intensity threshold
    fwhm = full width half maximum
    """
    #use daostarfinder to pick dots
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=None, exclude_border=True)
    sources = daofind(data)
    
    #return none if nothing was picked else return table
    if sources == None:
        return None
    for col in sources.colnames:
         sources[col].info.format = '%.8g'  # for consistent table output
    
    return sources

def get_optimum_fwhm(data, threshold):
    """
    Finds the best fwhm
    Parameters
    ----------
    data = 2D array
    threshold = initial threshold for testing
    """
    #generate fwhm to test
    fwhm_range = np.linspace(3,10,8)
    #get counts
    counts = []
    for fwhm in fwhm_range:
        try:
            dots = len(daofinder(data,  threshold, fwhm))
        except TypeError:
            continue
        counts.append(dots)
    #find index with largest counts
    if len(counts) == 0:
        #just return a number
        return 4
    else:
        best_index = np.argmax(counts)
        #this is the best fwhm
        best_fwhm = fwhm_range[best_index]

        return best_fwhm
    
def get_region_around(im, center, size, edge='raise'):
    """
    This function will essentially get a bounding box around detected dots
    
    Parameters
    ----------
    im = image tiff
    center = x,y centers from dot detection
    size = size of bounding box
    edge = "raise" will output error message if dot is at border and
            "return" will adjust bounding box 
            
    Returns
    -------
    array of boxed dot region
    """
    
    #calculate bounds
    lower_bounds = np.array(center) - size//2
    upper_bounds = np.array(center) + size//2 + 1
    
    #check to see if bounds is on edge
    if any(lower_bounds < 0) or any(upper_bounds > im.shape[-1]):
        if edge == 'raise':
            raise IndexError(f'Center {center} too close to edge to extract size {size} region')
        elif edge == 'return':
            lower_bounds = np.maximum(lower_bounds, 0)
            upper_bounds = np.minimum(upper_bounds, im.shape[-1])
    
    #slice out array of interest
    region = im[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]

    return region
    
def dot_detection(img, size_cutoff = 4, 
                  threshold = 300, channel = 1):
    
    """
    Perform dot detection on image using daostarfinder.
    
    Parameters
    ----------
    img = numpy array
    size_cutoff = number of standard deviation away from mean size area
    threshold = absolute pixel intensity the spot must be greater than
    channel = which channel to look at (1-4)
    
    Returns
    ----------
    locations csv file and size distribution plots
    """      
    
    #using daostarfinder detection
    if len(img.shape)==3:
        #get optimal fwhm
        fwhm = get_optimum_fwhm(img[channel-1], threshold=threshold)
        #dot detect
        peaks = daofinder(img[channel-1], threshold=threshold,fwhm=fwhm)
        #if None was returned then return empty df
        try:
            peaks = peaks.to_pandas()
        except AttributeError:
            return pd.DataFrame()
        peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
        ch = np.zeros(len(peaks))+channel
        z_slice = np.zeros(len(peaks))
        peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
        peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
        dots = peaks
    else:
        dots = []
        for z in range(img.shape[0]):
            #get optimal fwhm
            fwhm = get_optimum_fwhm(img[z][channel-1], threshold=threshold)
            #dot detect
            peaks = daofinder(img[z][channel-1], threshold=threshold, fwhm=fwhm)
            #if None was returned for a particular z then continue
            try:
                peaks = peaks.to_pandas()
            except AttributeError:
                continue
            peaks = peaks[["xcentroid" ,"ycentroid", "flux", "peak", "sharpness", "roundness1", "roundness2"]].values
            ch = np.zeros(len(peaks))+channel
            z_slice = np.zeros(len(peaks))+z
            peaks = np.append(peaks, ch.reshape(len(ch),1), axis=1)
            peaks = np.append(peaks, z_slice.reshape(len(z_slice),1), axis=1)
            dots.append(peaks)
        #check if combined df is empty
        if len(dots) == 0:
            return pd.DataFrame()
        else:
            dots = np.concatenate(dots)

    #make df and reorganize        
    dots = pd.DataFrame(dots)
    dots.columns = ["x", "y", "flux", "max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "ch", "z"]
    dots = dots[["ch","x","y","z", "flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits"]]

    #get area
    #subtract 1 from channels to get right slice
    coord = dots[["x","y","z","ch"]].values
    area_list = []
    for i in coord:
        x = int(i[0])
        y = int(i[1])
        z = int(i[2])
        c = int(i[3])
        #get bounding box
        try:
            if len(img.shape)==3:
                blob = get_region_around(img[c-1], center=[y,x], size=7, edge='raise')
            else:
                blob = get_region_around(img[z][c-1], center=[y,x], size=7, edge='raise')
        except IndexError:
            area_list.append(0)
            continue
        #estimate area of dot by local thresholding and summing binary mask
        try:
            local_thresh = threshold_local(blob, block_size=7)
            label_local = (blob > local_thresh)
            area = np.sum(label_local)
            area_list.append(area)
        except ValueError:
            area_list.append(0)
        except IndexError:
            area_list.append(0)

    #construct final df
    dots["size"] = area_list
    dots = dots[["ch","x","y","z","flux","max intensity", "sharpness", "symmetry", "roundness by gaussian fits", "size"]]

    if size_cutoff != None:
        #filter by size
        mu, std = norm.fit(dots["size"]) #fit gaussian to size dataset
        dots = dots[(dots["size"] < (mu+(size_cutoff*std))) 
                                & (dots["size"] > (mu-(size_cutoff*std)))]
    
    return dots
    
def quant_mask(img, mask, pos, channel1=1, channel2=None):
    
    """
    Function to quantify masks
    
    Parameters
    ----------
    img: img array
    mask: mask array
    pos: pos number
    channel1: which channel do you want to analyze 
    channel2: second channel you want to analyze (set to None if looking at one channel)
    """
    
    if channel2 == None:
        avg_int_list = []
        for cell in np.unique(mask):
            if cell != 0:
                area = np.sum((mask==cell))
                intensity = np.sum((mask==cell) * img[channel1-1])
                mean_int = intensity/area
                avg_int_list.append([f"cell{cell}_pos{pos}",mean_int])
        final = pd.DataFrame(avg_int_list)
        final.columns = ["Cell id", "Mean intensity"]
    else:
        avg_int_list = []
        for cell in np.unique(mask):
            if cell != 0:
                area = np.sum((mask==cell))
                intensity1 = np.sum((mask==cell) * img[channel1-1])
                intensity2 = np.sum((mask==cell) * img[channel2-1])
                mean_int1 = intensity1/area
                mean_int2 = intensity2/area
                avg_int_list.append([f"cell{cell}_pos{pos}",mean_int1, mean_int2])
        final = pd.DataFrame(avg_int_list)
        final.columns = ["Cell id", "Mean intensity 1", "Mean intensity 2"]

    return final

def linregress(x,y, no_intercept=False):
    """
    Linear regression function.
    Parameters
    ----------
    x: numpy array x
    y: numpy array y
    no_intercept: intercept at 0 or identify intercept
    """
    
    if no_intercept == False:
        #intercept not 0
        lm = LinearRegression()
        x=x.reshape(-1, 1)
        lm.fit(x, y)

        c = lm.intercept_
        m = lm.coef_
    else:
        #intercept forced to 0
        x_t = np.vstack([x, np.zeros(len(x))]).T
        m,c = np.linalg.lstsq(x_t, y, rcond=None)[0]
    
    return c,m

def gen_figure(df, time_course=True, no_intercept=True):
    """
    Function to generate scatter plots with fitted lines.
    Parameters
    ----------
    df: pandas dataframe
    time_course: bool on whether this is a time course
    """
    
    column_names = df.columns
    plt.figure(figsize=(4,5), dpi=300)
    
    if time_course == True:
        max_x = max(df[f"{column_names[1]}"])
        for time in df["Time"].unique():
            x = df[df["Time"] == time][f"{column_names[1]}"].values
            y = df[df["Time"] == time][f"{column_names[2]}"].values
            c,m = linregress(x,y, no_intercept=no_intercept)
            y_lin = [k*m+c for k in np.linspace(0,max_x,1000)]
            plt.scatter(x,y, s=7, alpha=0.5,marker="o")
            plt.plot(np.linspace(0,max_x,1000), y_lin, label=f"{time}", lw=1.5)
        plt.xlabel(f"{column_names[1]} Mean Intensity (a.u.)")
        plt.ylabel(f"{column_names[2]} Mean Intensity (a.u.)")
        sns.despine()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/{column_names[1]}_vs_{column_names[2]}.png", dpi=300)
    else:
        max_x = max(df[f"{column_names[1]}"])
        x = df[f"{column_names[1]}"].values
        y = df[f"{column_names[2]}"].values
        c,m = linregress(x,y, no_intercept=no_intercept)
        y_lin = [k*m+c for k in np.linspace(0,max_x,1000)]
        plt.scatter(x,y, s=7, alpha=0.5,marker="o")
        plt.plot(np.linspace(0,max_x,1000), y_lin, lw=1.5)
        plt.xlabel(f"{column_names[1]} Mean Intensity (a.u.)")
        plt.ylabel(f"{column_names[2]} Mean Intensity (a.u.)")
        sns.despine()
        plt.tight_layout()
        plt.savefig(f"./results/{column_names[1]}_vs_{column_names[2]}.png", dpi=300)
        
    return plt