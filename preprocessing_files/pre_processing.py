"""
authors: Katsuya Lex Colon and Shaan Sekhon
updated: 12/10/21
group: Cai Lab
"""
#image processing
import cv2
from skimage import util
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from scipy import ndimage
from skimage.exposure import adjust_gamma
from sklearn import linear_model
from skimage import restoration
from skimage.feature import peak_local_max
from skimage.exposure import match_histograms
#general analysis packages
import numpy as np
#directory management
import os
from pathlib import Path
#plotting packages
import matplotlib.pyplot as plt
import plotly.express as px
#image reading
import tifffile as tf
#psf
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import AiryDisk2DKernel
#extra
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

def background_correct_image(stack, correction_algo, stack_bkgrd=None, z=2, size=2048, 
                             gamma = 1.4, kern=5, sigma=40, match_hist =True, subtract=True, divide=False):
    '''
   This function will background correct raw images. There are several correction algorithms that can be used (SigmaClipping_and_Gamma_C,Gaussian_and_Gamma_Correction, and LSR_Backgound_Correction).
   Additionally, one can choose to use final or initial background image for subtraction if stack_bkgrd (background image array) is provided. 
   Parameters
   ----------
   stack = raw image
   correction_algo = SigmaClipping_and_Gamma_C,Gaussian_and_Gamma_Correction, and LSR_Backgound_Correction
   stack_bkgrd = initial or final background image array
   z = number of z slices
   size = image size
   gamma = gamma enhancment values
   kern = kernel size
   sigma = sigma value for gaussian blurring
   match_hist = bool to match histograms of blurred image
   subtract = bool to subtract blurred image from raw
   divide = bool to divide blurred image from raw
    
    '''
    #check z's
    if len(stack.shape) == 3:
        channels = stack.shape[0]
        stack = stack.reshape(z,channels,size,size)
        if type(stack_bkgrd) != type(None):
            stack_bkgrd = stack_bkgrd.reshape(z,channels,size,size)
    
    if type(stack_bkgrd) != type(None):
        #only subtract non dapi channels
        len_ch = stack.shape[1]
        stack_sub = util.img_as_int(stack[:,:len_ch-1,:,:])-util.img_as_int(stack_bkgrd[:,:len_ch-1,:,:])
        stack_sub[stack_sub<0]=0
        #add back dapi
        dapi = stack[:,len_ch-1,:,:].reshape(z,1,size,size)
        stack= np.concatenate((stack_sub,dapi),1)
    else:
        stack=stack
        
    corrected_stack = []
    for z_slice in stack:
        corrected_z_slice = []
        for i in range(z_slice.shape[0]):
            #get specific channel from a z slice
            channel = z_slice[i]
            channel = np.asarray([float(i) for i in channel.flatten()]).reshape(channel.shape)
            #apply correction
            if correction_algo != LSR_Backgound_Correction:
                if i == (z_slice.shape[0]-1):
                    correction_algo = LSR_Backgound_Correction
                    corrected_channel = correction_algo(channel)
                else:
                    if correction_algo == Gaussian_and_Gamma_Correction:
                        corrected_channel = correction_algo(channel, gamma, sigma, kern,
                                                            match_hist, subtract, divide)
                    else:
                        corrected_channel = correction_algo(channel, gamma)
                corrected_channel = np.asarray([np.uint16(i) for i in corrected_channel.flatten()]).reshape(corrected_channel.shape)
                corrected_z_slice.append(corrected_channel)
            else:
                corrected_channel = correction_algo(channel)
                corrected_channel = np.asarray([np.uint16(i) for i in corrected_channel.flatten()]).reshape(corrected_channel.shape)
                corrected_z_slice.append(corrected_channel)
        corrected_stack.append(corrected_z_slice)
    return np.array(corrected_stack)

#def tophat_background(image, kern=3):
#    """tophat raw then initial background subtraction"""
#    Getting the kernel to be used in Top-Hat
#    filterSize =(kern, kern)
#    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
#    Applying the Top-Hat operation
#    tophat_img = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
#    
#    return tophat_img

def SigmaClipping_and_Gamma_C(image, gamma):
    """Background Correction via Background Elimination 
    (estimated by sigma-clipped background), followed by Gamma Correction"""
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (30,30), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    sigma_clipped = image/(bkg.background/np.mean(bkg.background))
    sigma_clipped_contrast_ench_Gamma_c = adjust_gamma(sigma_clipped, gamma)
    sigma_clipped_contrast_ench_Gamma_c = (sigma_clipped_contrast_ench_Gamma_c/np.mean(sigma_clipped_contrast_ench_Gamma_c))*np.mean(image)
    
    return sigma_clipped_contrast_ench_Gamma_c

def Gaussian_and_Gamma_Correction(image, gamma, sigma, kern = 5, match_hist =True, subtract=True, divide=False):
    """ Background Correction via Background Elimination (estimated by Gaussian), followed by Gamma Correction"""
    #2d gaussian blur image
    image_blur = cv2.GaussianBlur(image,(kern,kern),sigma)
    if divide == False:
        #subtract gaussian blurred background
        if (match_hist == True) and (subtract == True):
            image_background_eliminated= image - match_histograms(image_blur, image)
            image_background_eliminated[image_background_eliminated<0] = 0 
        elif (match_hist == False) and (subtract == True):
            image_background_eliminated= image - image_blur
            image_background_eliminated[image_background_eliminated<0] = 0 
    else:
        #1d convolution is better for evening out background
        image_blur = ndimage.gaussian_filter(image, sigma)
        #divide gaussian blurred background to even out illumination
        image_background_eliminated= image/((image_blur)/np.mean(image_blur))
    #adjust contrast by gamma enhancement
    contrast_ench_Gamma_c = adjust_gamma(image_background_eliminated, gamma)
    #rescale image by constant factor
    contrast_ench_Gamma_c = (contrast_ench_Gamma_c/np.mean(contrast_ench_Gamma_c))*100
    
    return contrast_ench_Gamma_c

def LSR_Backgound_Correction(image):
    """Edge-Dimming Correction via Least-Squares Regression"""
    xx, yy = np.meshgrid(np.arange(0, image.shape[0]),np.arange(0, image.shape[1]))
    x_vals,y_vals = xx.flatten(),yy.flatten()
    center_x,center_y = int(image.shape[0]/2),int(image.shape[1]/2)
    dist_from_center = (((x_vals - center_x)**2) + ((y_vals - center_y)**2))**(1/2)
    model = linear_model.LinearRegression()
    model.fit(np.asarray(dist_from_center)[:,np.newaxis], np.asarray(image.flatten())[:,np.newaxis])
    pred = (dist_from_center*model.coef_[0][0])+model.intercept_[0]
    
    image_corrected =  image.flatten()/(pred/np.mean(pred))
    return image_corrected.reshape(image.shape)

def remove_fiducials(image_ref, tiff, size=9,min_distance=10,threshold_abs=1000,
                       num_peaks=1000, edge='raise'):
    """
    This function will create a square mask around each fiducial and use the masks to eliminate the fiducial.
    
    Parameters
    ----------
    im = image tiff
    center = x,y centers from dot detection
    size = size of bounding box
    normalize = bool to normalize intensity
    edge = "raise" will output error message if dot is at border and
            "return" will adjust bounding box 
            
    Returns
    -------
    fiducial subtracted image
    """
    cand_list = []
    if len(image_ref.shape) == 3:
        #pick out bright dots
        for c in range(image_ref.shape[0]):
            cands_ref = peak_local_max(image_ref[c], min_distance=min_distance, 
                           threshold_abs=threshold_abs, num_peaks=num_peaks)
            cand_list.append(cands_ref)
    else:
        #pick out bright dots
        for z in range(image_ref.shape[0]):
            z_slice = []
            for c in range(image_ref.shape[1]):
                cands_ref = peak_local_max(image_ref[z][c], min_distance=min_distance, 
                               threshold_abs=threshold_abs, num_peaks=num_peaks)
                z_slice.append(cands_ref)
            cand_list.append(z_slice)
    
    #copy image
    mask_ref = image_ref.copy()
    mask_stack = []
    if len(image_ref.shape) == 3:
        for c in range(len(cand_list)):
            mask_c = mask_ref[c]
            for dots in cand_list[c]:
                #calculate bounds
                lower_bounds = np.array(dots).astype(int) - size//2
                upper_bounds = np.array(dots).astype(int) + size//2 + 1

                #check to see if bounds is on edge
                if any(lower_bounds < 0) or any(upper_bounds > image_ref.shape[-1]):
                    if edge == 'raise':
                        raise IndexError(f'Center {center} too close to edge to extract size {size} region')
                    elif edge == 'return':
                        lower_bounds = np.maximum(lower_bounds, 0)
                        upper_bounds = np.minimum(upper_bounds, image_ref.shape[-1])

                #convert region into 1
                mask_c[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]=True
            mask_stack.append(mask_c)
    else:
        for z in range(len(cand_list)):
            mask_z = []
            for c in range(len(cand_list[z])):
                mask_c = mask_ref[z][c]
                for dots in cand_list[z][c]:
                    #calculate bounds
                    lower_bounds = np.array(dots).astype(int) - size//2
                    upper_bounds = np.array(dots).astype(int) + size//2 + 1

                    #check to see if bounds is on edge
                    if any(lower_bounds < 0) or any(upper_bounds > image_ref.shape[-1]):
                        if edge == 'raise':
                            raise IndexError(f'Center {center} too close to edge to extract size {size} region')
                        elif edge == 'return':
                            lower_bounds = np.maximum(lower_bounds, 0)
                            upper_bounds = np.minimum(upper_bounds, image_ref.shape[-1])

                    #convert region into 1
                    mask_c[lower_bounds[0]:upper_bounds[0], lower_bounds[1]:upper_bounds[1]]=True
                mask_z.append(mask_c)
            mask_stack.append(mask_z)
            
    #make bool mask
    mask_stack = np.array(mask_stack)
    final = (mask_stack==True)
    #invert mask
    final_inv = (final==False).astype(int)
    
    #elementwise multiplication of arrays
    new_image = (tiff * final_inv).astype(np.uint16)
    
    #return mask
    return new_image
    
def gen_psf(model="gaussian", sigma=2, radius=6, size=7):
    
    """
    A general function to return model point spread functions.
    
    Parameters
    ----------
    model: "gaussian" or "airy_disc"
    sigma: standard deviation on x and y for gaussian psf
    radius: radius of central disc for the airy disc psf
    size: size of kernel
    """
    
    if model == "gaussian":
        gaussian_2D_kernel = Gaussian2DKernel(x_stddev=sigma,y_stddev=sigma,x_size=size,y_size=size, mode='oversample')
        return np.array(gaussian_2D_kernel)
    if model == "airy_disc":
        airy_disc = AiryDisk2DKernel(radius=radius, x_size=size, y_size=size, mode = 'oversample')
        return np.array(airy_disc)
    
def RL_deconvolution(image, kern_rl=7, sigma=(1.8,1.6,1.5,1.3), 
                   radius=(4,4,4,4), model="gaussian", microscope = "boc"):
    """Assuming a gaussian or airy disc psf, images are deconvoluted using the richardson-lucy algorithm
    Parameters
    ----------
    image = multi or single array of images (z,c,x,y)
    kern_rl = kernel size
    sigma = define sigma at each channel (750nm,647nm,555nm,488nm)
    radius = define radius of airy disc at each channel (750nm,647nm,555nm,488nm)
    model = "gaussian" or "airy_disc" psf
    microscope = use preset sigmas for defined scope ("boc" and "lb")
    """
    # defined sigma from testing
    sigma_dict = {"boc":[1.8,1.6,1.5,1.3],"lb":[2.0,1.7,1.3,1.2]}
    
    #check to see if it is one z
    if len(image.shape) == 3:
        channel_slice = []
        #perform deconvolution on each channel
        if microscope == None:
            for c in range(image.shape[0]):
                psf = gen_psf(model=model, sigma=sigma[c], radius=radius[c], size=kern_rl)
                adj_img = util.img_as_float(image[c]) + 1E-4
                deconvolved_RL = restoration.richardson_lucy(adj_img, psf, 10)
                channel_slice.append(deconvolved_RL)
            return util.img_as_uint(np.array(channel_slice)), psf
        else:
            ch_sigma = sigma_dict[microscope]
            for c in range(image.shape[0]):
                psf = gen_psf(model=model, sigma=ch_sigma, radius=radius[c], size=kern_rl)
                adj_img = util.img_as_float(image[c]) + 1E-4
                deconvolved_RL = restoration.richardson_lucy(adj_img, psf, 10)
                channel_slice.append(deconvolved_RL)
            return util.img_as_uint(np.array(channel_slice)), psf
    
    else:
        if microscope == None:
            z_slice=[] 
            #go across z's and channels
            for z in range(image.shape[0]):
                channel_slice=[]
                #deconvolution
                for c in range(image.shape[1]):
                    psf = gen_psf(model=model, sigma=sigma[c], radius=radius[c], size=kern_rl)
                    adj_img = util.img_as_float(image[z][c]) + 1E-4
                    deconvolved_RL = restoration.richardson_lucy(adj_img, psf, 10)
                    channel_slice.append(deconvolved_RL)
                z_slice.append(channel_slice)
            img_arr = np.array(z_slice)
            img_arr = util.img_as_uint(img_arr)
            del z_slice
            return img_arr, psf
        else:
            ch_sigma = sigma_dict[microscope]
            z_slice=[] 
            #go across z's and channels
            for z in range(image.shape[0]):
                channel_slice=[]
                #deconvolution
                for c in range(image.shape[1]):
                    psf = gen_psf(model=model, sigma=ch_sigma[c], radius=radius[c], size=kern_rl)
                    adj_img = util.img_as_float(image[z][c]) + 1E-4
                    deconvolved_RL = restoration.richardson_lucy(adj_img, psf, 10)
                    channel_slice.append(deconvolved_RL)
                z_slice.append(channel_slice)
            img_arr = np.array(z_slice)
            img_arr = util.img_as_uint(img_arr)
            del z_slice
            return img_arr, psf
        
def low_pass_gaussian(image, kern=3):
    """A low pass gaussian blur
    Parameters
    ----------
    image = single or list of arrays
    kern = int
    """
    if len(image.shape) == 3:
        c_slice = []
        for c in range(image.shape[0]):
            c_slice.append(cv2.GaussianBlur(image[c],(kern,kern),1))
        return np.array(c_slice)
    else:
        z_slice = []
        for z in range(image.shape[0]):
            channel_slice = []
            for c in range(image.shape[1]):
                channel_slice.append(cv2.GaussianBlur(image[z][c],(kern,kern),1))
            z_slice.append(channel_slice)
        return np.array(z_slice)

def deconvolute_one(image_path, image_ref, sigma_hpgb = 1, kern_hpgb=5, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    size=9,min_distance=10,threshold_abs=1000,
                    num_peaks=1000, gamma=1, hyb_offset=0, edge='raise', swapaxes=True,
                    noise= True, bkgrd_sub=True, remove_fiducial=False, 
                    match_hist=True, subtract=True, divide=False):
    
    """deconvolute one image only
    Parameters
    ----------
    image_path = path to single image
    image_ref = path to reference image for removing fiducials
    sigma_hpgb = sigma for high-pass gaussian blur
    kern_rl = kernel size for Richardson-Lucy deconvolution
    kern_lpgb = kernel size for low-pass gaussian blur
    kern_hpgb = kernel size for high-pass gaussian blur
    sigma = channel specific sigma values for RL-deconvolution (not used if microscope is defined)
    radius = channel specific radius for airy disc psf (fill always)
    model = "gaussian" or "airy_disc" psf
    microscope = which scope you used (only have leica boss or box of chocolates)
    size= bounding box size for remove fiducials
    min_distance = number of pixels to peaks need to be away for remove fiducial function
    threshold_abs = absolute threshold used in remove fiducial function
    num_peaks = number of total dots for remove fiducials
    gamma = interge for gamma enhancement
    hyb_offset = number of hybcycles to subtract for file name adjustment
    edge = argument for bounding box in remove fiducials
    swapaxes = bool to swap axes when reading in an image
    noise = re-convolve image at the end
    bkgrd_sub = bool to perform background subtraction
    remove_fiducial = bool to keep or remove fiducials
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    """
    
    #make output directory
    orig_image_dir = Path(image_path).parent.parent
    output_folder = Path(orig_image_dir).with_name('deconvoluted_images')
    output_path = output_folder / Path(image_path).relative_to(orig_image_dir)
    if hyb_offset != 0:
        hyb_number = int(output_path.parent.name.split("_")[1])
        new_number = "HybCycle_" + str(hyb_number-offset)
        pos_name = Path(image_path).name
        output_path = output_path.parent.parent / new_number /pos_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read in image
    if swapaxes == True:
        image = tf.imread(image_path)
        image = np.swapaxes(image,0,1)
        if bkgrd_sub == True:
            bkgrd_tiff_src =  Path(image_path).parent.parent / "final_background"
            pos = Path(image_path).name
            stack_bkgrd_path = bkgrd_tiff_src / pos
            stack_bkgrd = tf.imread(stack_bkgrd_path)
            stack_bkgrd = np.swapaxes(stack_bkgrd,0,1)
        else:
            stack_bkgrd=None
    else:
        image = tf.imread(image_path)
        if bkgrd_sub == True:
            bkgrd_tiff_src =  Path(image_path).parent.parent / "final_background"
            pos = Path(image_path).name
            stack_bkgrd_path = bkgrd_tiff_src / pos
            stack_bkgrd = tf.imread(stack_bkgrd_path)
        else:
            stack_bkgrd=None
    #remove fiducials       
    if remove_fiducial == True:
        ref_image = tf.imread(image_ref)
        fid_rem_image = remove_fiducials(ref_image, image, size=size,
                                         min_distance=min_distance,threshold_abs=threshold_abs,
                                         num_peaks=num_peaks, edge='raise')
        image = fid_rem_image.copy()
        
    #perform background correction    
    if len(image.shape) ==4:
        z=image.shape[0]
        size = image.shape[2]
        print('background correction...')
        hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, 
                                              stack_bkgrd, z, size, gamma, kern_hpgb, sigma_hpgb,
                                              match_hist, subtract, divide) 
    else:
        z=1
        size = image.shape[2]
        print('background correction...')
        hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, 
                                              stack_bkgrd, z, size,gamma,kern_hpgb, sigma_hpgb,
                                              match_hist, subtract, divide) 
    #perform deconvolution
    print('deconvolution...')
    if len(image.shape) ==3:
        rl_img_hpgb,psf = RL_deconvolution(hpgb_image[:4,:,:], kern_rl = kern_rl,
                                            sigma=sigma, radius=radius,model=model,
                                            microscope = microscope)
    else:
        rl_img_hpgb,psf = RL_deconvolution(hpgb_image[:,:4,:,:], kern_rl = kern_rl,
                                            sigma=sigma, radius=radius,model=model,
                                            microscope = microscope)
    #reconvolve and write
    if noise == True:
        print('low pass gaussian and writing image...')
        lpgb = low_pass_gaussian(rl_img_hpgb, kern = kern_lpgb)
        tf.imwrite(output_path, lpgb)
    else:
        print('writing image')
        tf.imwrite(output_path, rl_img_hpgb)
            
def deconvolute_many(images, image_ref, sigma_hpgb = 1, kern_hpgb = 5, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    size=9,min_distance=10,threshold_abs=1000,
                    num_peaks=1000, gamma = 1, hyb_offset=0, edge='raise', swapaxes=True,
                    noise= True, bkgrd_sub=True, remove_fiducial=False, 
                    match_hist=True, subtract=True, divide=False):
    
    """function to deconvolute all images
     Parameters
    ----------
    images = list of image paths
    image_ref = path to reference image for removing fiducials
    sigma_hpgb = sigma for high-pass gaussian blur
    kern_hpgb = kernel size for high-pass gaussian blur
    kern_rl = kernel size for Richardson-Lucy deconvolution
    kern_lpgb = kernel size for low-pass gaussian blur
    sigma = channel specific sigma values for RL-deconvolution (not used if microscope is defined)
    radius = channel specific radius for airy disc psf (fill always)
    model = "gaussian" or "airy_disc" psf
    microscope = which scope you used (only have leica boss or box of chocolates)
    size= bounding box size for remove fiducials
    min_distance = number of pixels to peaks need to be away for remove fiducial function
    threshold_abs = absolute threshold used in remove fiducial function
    num_peaks = number of total dots for remove fiducials
    gamma = interger for gamma enhancement
    edge = argument for bounding box in remove fiducials
    swapaxes = bool to swap axes when readin in an image
    noise = re-convolve image at the end
    bkgrd_sub = bool to perform background subtraction
    remove_fiducial = bool to keep or remove fiducials
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    """
    
    import time
    start = time.time()
    
    if type(images) != list:
        deconvolute_one(images, image_ref, 
                       sigma_hpgb=sigma_hpgb,kern_hpgb=kern_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       size=size,min_distance=min_distance,threshold_abs=threshold_abs,gamma=gamma,hyb_offset=hyb_offset,
                       num_peaks=num_peaks, edge=edge, swapaxes=swapaxes,
                       noise=noise, bkgrd_sub=bkgrd_sub, remove_fiducial=remove_fiducial, 
                       match_hist=match_hist, subtract=subtract, divide=divide)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(deconvolute_one, path, image_ref, sigma_hpgb=sigma_hpgb,kern_hpgb=kern_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       size=size,min_distance=min_distance,threshold_abs=threshold_abs,gamma=gamma,hyb_offset=hyb_offset,
                       num_peaks=num_peaks, edge=edge, swapaxes=swapaxes,
                       noise=noise, bkgrd_sub=bkgrd_sub, remove_fiducial=remove_fiducial, 
                       match_hist=match_hist, subtract=subtract, divide=divide)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
                
def bkgrd_corr_one(image_path, correction_type = None, stack_bkgrd=None, swapaxes=False, 
                   z=2, size=2048, gamma = 1.4, kern_hpgb=5, sigma = 40, rb_radius=5, hyb_offset=0,
                   rollingball = False, lowpass=True, match_hist=True, subtract=True, divide=False):
    """
    background correct one image only
    
    Parameters
    ----------
    image_path= path to image
    correction_type = which correction algo to use
    stack_bkgrd = 4 or 3d array of background image
    swapaxes=bool to swapaxes
    z=number of z
    size=x and y shape
    gamma = int for gamma enhancement
    kern_hpgb = kernel size for hpgb 
    sigma = number of sigmas for hpgb
    rb_radius = rolling ball size
    hyb_offset = number of hybcycles to subtract for file name adjustment
    lowpass = do a low pass gaussian filter
    rollingball = do a rolling ball subtraction
    lowpass = bool to blur image with gaussian
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    """
   
    orig_image_dir = Path(image_path).parent.parent
    output_folder = Path(orig_image_dir).with_name('pre_processed_images')
    output_path = output_folder / Path(image_path).relative_to(orig_image_dir)
    if hyb_offset != 0:
        hyb_number = int(output_path.parent.name.split("_")[1])
        new_number = "HybCycle_" + str(hyb_number-offset)
        pos_name = Path(image_path).name
        output_path = output_path.parent.parent / new_number /pos_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if swapaxes == True:
        image = tf.imread(image_path)
        image = np.swapaxes(image,0,1)
        if type(stack_bkgrd) != type(None):
            bkgrd = tf.imread(stack_bkgrd)
            bkgrd = np.swapaxes(bkgrd,0,1)
    else:
        image = tf.imread(image_path)
        if len(image.shape) == 3:
            channels = image.shape[0]
            image = image.reshape(z,channels,size,size)
        if type(stack_bkgrd) != type(None):
            bkgrd = tf.imread(stack_bkgrd)
            if len(bkgrd.shape)==3:
                bkgrd = bkgrd.reshape(z,channels,size,size)
    
    #background correct
    if type(stack_bkgrd) != type(None):
        corr_img = background_correct_image(image, correction_type, bkgrd, 
                                            z, size, gamma, kern_hpgb, sigma, match_hist, subtract, divide)
    else:
        corr_img = background_correct_image(image, correction_type, stack_bkgrd,
                                            z, size, gamma, kern_hpgb, sigma, match_hist, subtract, divide)
    
    #do rolling ball subtraction
    if rollingball == True:
        img_stack = []
        for z in range(corr_img.shape[0]):
            c_stack = []
            for c in range(corr_img.shape[1]):
                background = restoration.rolling_ball(corr_img[z][c], radius=rb_radius)
                rb_img = corr_img[z][c]-background
                rb_img[rb_img<0]=0
                c_stack.append(rb_img)
            img_stack.append(c_stack)
        corr_img = np.array(img_stack)
        
    #low pass filter
    if lowpass == True:
        print('low pass gaussian and writing image...')
        lpgb = low_pass_gaussian(corr_img, kern = 3)
        tf.imwrite(str(output_path), lpgb)
    else:
        print('writing image')
        tf.imwrite(str(output_path), corr_img)

def correct_many(images, correction_type = None, stack_bkgrd=None, swapaxes=False,
                 z=2, size=2048, gamma = 1.4,kern_hpgb=5, sigma=40, rb_radius=5, hyb_offset=0,
                 rollingball=False, lowpass = True, match_hist=True, subtract=True, divide=False):
    """
    function to correct all image
    
    Parameters
    ----------
    images = path to multiple images
    correction_type = which correction algo to use
    stack_bkgrd = 4 or 3d array of background image
    swapaxes=bool to swapaxes
    z=number of z
    size=x and y shape
    gamma=int
    sigma = int
    rollingball = do a rolling ball subtraction
    lowpass = do a low pass gaussian filter 
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    """
    import time
    start = time.time()
    
    if type(images) != list:
        bkgrd_corr_one(images, correction_type,stack_bkgrd, swapaxes, z, size,  
                       gamma, kern_hpgb,sigma, rb_radius, hyb_offset, rollingball, 
                       lowpass,  match_hist, subtract, divide)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(bkgrd_corr_one, path, correction_type, stack_bkgrd,
                                 swapaxes, z, size,  gamma,kern_hpgb, sigma, rb_radius,hyb_offset,
                                 rollingball, lowpass,  match_hist, subtract, divide)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
