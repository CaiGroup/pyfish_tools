"""
authors: Katsuya Lex Colon, Shaan Sekhon, and Anthony Linares
updated: 06/20/22
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
from skimage.exposure import match_histograms
#general analysis packages
import numpy as np
#directory management
import os
from pathlib import Path
#image reading
import tifffile as tf
from util import pil_imread
#psf
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import AiryDisk2DKernel
#extra
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")

def background_correct_image(stack, correction_algo, stack_bkgrd=None, 
                             gamma = 1.4, kern=5, sigma=40, match_hist =True, 
                             subtract=True, divide=False, tophat_raw=False):
    '''
   This function will background correct raw images. There are several correction algorithms that can be used (SigmaClipping_and_Gamma_C,Gaussian_and_Gamma_Correction, and LSR_Backgound_Correction).
   Additionally, one can choose to use final or initial background image for subtraction if stack_bkgrd (background image array) is provided. 
   Parameters
   ----------
   stack = raw image
   correction_algo = SigmaClipping_and_Gamma_C,Gaussian_and_Gamma_Correction, and LSR_Backgound_Correction
   stack_bkgrd = initial or final background image array
   gamma = gamma enhancment values
   kern = kernel size
   sigma = sigma value for gaussian blurring
   match_hist = bool to match histograms of blurred image
   subtract = bool to subtract blurred image from raw
   divide = bool to divide blurred image from raw
   tophat_raw = bool to perform tophat on raw image
    
    '''
    #check z's
    if len(stack.shape) == 3:
        channels = stack.shape[0]
        size = stack.shape[1]
        stack = stack.reshape(1,channels,size,size)
        if type(stack_bkgrd) != type(None):
            stack_bkgrd = stack_bkgrd.reshape(1,channels,size,size)
    
    #run tophat on raw image if desired
    if tophat_raw == True:
        stack = tophat_image(stack)
    
    #perform background subtraction on image using actual initial or final background image
    if type(stack_bkgrd) != type(None):
        #only subtract non dapi channels
        len_ch = stack.shape[1]
        size = stack.shape[2]
        stack_sub = util.img_as_int(stack[:,:len_ch-1,:,:])-util.img_as_int(stack_bkgrd[:,:len_ch-1,:,:])
        stack_sub[stack_sub<0]=0
        #add back dapi
        dapi = stack[:,len_ch-1,:,:].reshape(stack.shape[0],1,size,size)
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
                    correction_algo_dapi = LSR_Backgound_Correction
                    corrected_channel = correction_algo_dapi(channel)
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

def tophat_image(stack):
    """
    Tophat raw image to help removes large blobs (like lipofusin). 
    """
    #kernel size for tophat (5 is good for lipofusin)
    tophat_kernel_size = 5
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(tophat_kernel_size), int(tophat_kernel_size)))
    
    #create empty array
    tophat_stack = np.zeros(stack.shape)
    #get dapi channel
    dapi_ch = stack.shape[1]-1
    
    #go through z and channels
    for z in range(stack.shape[0]):
        for ch in range(stack.shape[1]-1):
            tophat_stack[z][ch] = cv2.morphologyEx(stack[z][ch], cv2.MORPH_TOPHAT, tophat_kernel)
        tophat_stack[z][dapi_ch] = stack[z][dapi_ch]
        
    #reconvert to uint16
    stack = tophat_stack.astype('uint16')   
    
    return stack

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
    
    if (divide == False) and (subtract == True):
        #2d gaussian blur image
        image_blur = cv2.GaussianBlur(image,(kern,kern),sigma)
        #subtract gaussian blurred background
        if (match_hist == True) and (subtract == True):
            image_background_eliminated= image - match_histograms(image_blur, image)
            image_background_eliminated[image_background_eliminated<0] = 0 
        elif (match_hist == False) and (subtract == True):
            image_background_eliminated= image - image_blur
            image_background_eliminated[image_background_eliminated<0] = 0 
            
    elif (divide == True) and (subtract == True):
        #1d convolution is better for evening out background
        image_blur = ndimage.gaussian_filter(image, 30)
        #divide gaussian blurred background to even out illumination
        image_background_even= image/((image_blur)/np.mean(image_blur))
        #2d gaussian blur image
        image_blur = cv2.GaussianBlur(image_background_even,(kern,kern),sigma)
        if (match_hist == True) and (subtract == True):
            image_background_eliminated= image - match_histograms(image_blur, image)
            image_background_eliminated[image_background_eliminated<0] = 0 
        elif (match_hist == False) and (subtract == True):
            image_background_eliminated= image - image_blur
            image_background_eliminated[image_background_eliminated<0] = 0 
    else:
        #1d convolution is better for evening out background
        image_blur = ndimage.gaussian_filter(image, 30)
        #divide gaussian blurred background to even out illumination
        image_background_eliminated= image/((image_blur)/np.mean(image_blur))
    #adjust contrast by gamma enhancement
    contrast_ench_Gamma_c = adjust_gamma(image_background_eliminated, gamma)
    
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

def scale_int(image, p_min=80, p_max=100):
    """
    This function will scale the intensities for each position. The following code was adapted from starfish.
    Parameters
    ----------
    image = image stack (z,c,x,y)
    p_min = minimum percentile
    p_max = maximum percentile
    """
    #check the shape of image
    if len(image.shape)==3:
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
     
    final_stack = []
    for z in range(image.shape[0]):
        z_stack = []
        for c in range(image.shape[1]):
            #slice out image
            img_slice = image[z,c,:,:]
            #get the min and max percentile values
            v_min, v_max = np.percentile(img_slice, [p_min, p_max])
            #clip and set min to 0
            image_set_zero = img_slice.clip(min=v_min, max=v_max) - np.float32(v_min)
            #scale image to max value
            max_int = np.max(image_set_zero)
            image_scaled = image_set_zero/max_int
            z_stack.append(image_scaled)
        final_stack.append(z_stack)
    
    #final image
    scaled_stack = np.array(final_stack)
    
    return scaled_stack

def deconvolute_one(image_path, sigma_hpgb = 1, kern_hpgb=5, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    gamma=1, hyb_offset=0, num_channels=4,
                    noise= True, bkgrd_sub=True, 
                    match_hist=True, subtract=True, divide=False, tophat_raw=False):
    
    """deconvolute one image only
    Parameters
    ----------
    image_path = path to single image
    sigma_hpgb = sigma for high-pass gaussian blur
    kern_rl = kernel size for Richardson-Lucy deconvolution
    kern_lpgb = kernel size for low-pass gaussian blur
    kern_hpgb = kernel size for high-pass gaussian blur
    sigma = channel specific sigma values for RL-deconvolution (not used if microscope is defined)
    radius = channel specific radius for airy disc psf (fill always)
    model = "gaussian" or "airy_disc" psf
    microscope = which scope you used (only have leica boss or box of chocolates)
    gamma = interge for gamma enhancement
    hyb_offset = number of hybcycles to subtract for file name adjustment
    num_channels = number of channels in image
    noise = re-convolve image at the end
    bkgrd_sub = bool to perform background subtraction
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    tophat_raw = bool to perform tophat on raw image
    """
    
    #make output directory
    parent = Path(image_path).parent
    while "seqFISH_datapipeline" not in os.listdir(parent):
        parent = parent.parent
    output_dir = parent / "seqFISH_datapipeline" /"output"/ "deconvoluted_images"
    hybcycle =  Path(image_path).parent.name
    output_path = output_dir / hybcycle / Path(image_path).name
    
    if hyb_offset != 0:
        hyb_number = int(output_path.parent.name.split("_")[1])
        new_number = "HybCycle_" + str(hyb_number-hyb_offset)
        pos_name = Path(image_path).name
        output_path = output_path.parent.parent / new_number /pos_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read in image
    image = pil_imread(image_path, swapaxes=True)
    if image.shape[1] != num_channels:
        image = pil_imread(image_path, swapaxes=False)
    
    if bkgrd_sub == True:
        bkgrd_tiff_src =  Path(image_path).parent.parent / "final_background"
        pos = Path(image_path).name
        stack_bkgrd_path = bkgrd_tiff_src / pos
        stack_bkgrd = pil_imread(stack_bkgrd_path, swapaxes=True)
        if stack_bkgrd.shape[1] != num_channels:
           stack_bkgrd = pil_imread(stack_bkgrd_path, swapaxes=False) 
    else:
        stack_bkgrd=None

        
    #perform background correction    
    if len(image.shape) ==4:
        print('background correction...')
        hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, 
                                              stack_bkgrd, gamma, kern_hpgb, sigma_hpgb,
                                              match_hist, subtract, divide, tophat_raw) 
    else:
        print('background correction...')
        hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, 
                                              stack_bkgrd, gamma,kern_hpgb, sigma_hpgb,
                                              match_hist, subtract, divide, tophat_raw) 
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
            
def deconvolute_many(images, sigma_hpgb = 1, kern_hpgb = 5, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    gamma = 1, hyb_offset=0, num_channels=4,
                    noise= True, bkgrd_sub=True, 
                    match_hist=True, subtract=True, divide=False, tophat_raw=False):
    
    """function to deconvolute all images
     Parameters
    ----------
    images = list of image paths
    sigma_hpgb = sigma for high-pass gaussian blur
    kern_hpgb = kernel size for high-pass gaussian blur
    kern_rl = kernel size for Richardson-Lucy deconvolution
    kern_lpgb = kernel size for low-pass gaussian blur
    sigma = channel specific sigma values for RL-deconvolution (not used if microscope is defined)
    radius = channel specific radius for airy disc psf (fill always)
    model = "gaussian" or "airy_disc" psf
    microscope = which scope you used (only have leica boss or box of chocolates)
    gamma = interger for gamma enhancement
    num_channels = number of channels in image
    noise = re-convolve image at the end
    bkgrd_sub = bool to perform background subtraction
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    tophat_raw = bool to perform tophat on raw image
    """
    
    import time
    start = time.time()
    
    if type(images) != list:
        deconvolute_one(images,
                       sigma_hpgb=sigma_hpgb,kern_hpgb=kern_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       gamma=gamma,hyb_offset=hyb_offset,
                       num_channels=num_channels,
                       noise=noise, bkgrd_sub=bkgrd_sub, 
                       match_hist=match_hist, subtract=subtract, divide=divide, tophat_raw=tophat_raw)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(deconvolute_one, path,sigma_hpgb=sigma_hpgb,kern_hpgb=kern_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       gamma=gamma,hyb_offset=hyb_offset,
                       num_channels=num_channels,noise=noise, bkgrd_sub=bkgrd_sub, 
                       match_hist=match_hist, subtract=subtract, divide=divide, tophat_raw=tophat_raw)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
                              
def bkgrd_corr_one(image_path, correction_type = None, stack_bkgrd=None, num_channels=4,
                   gamma = 1.4, kern_hpgb=5, sigma = 40, rb_radius=5, hyb_offset=0, p_min=80,
                   p_max = 99.999, norm_int = True, rollingball = False, 
                   lowpass=True, match_hist=True, subtract=True, divide=False, tophat_raw=False):
    """
    background correct one image only
    
    Parameters
    ----------
    image_path= path to image
    correction_type = which correction algo to use
    stack_bkgrd = 4 or 3d array of background image
    num_channels = number of expected channels for axes checking
    gamma = int for gamma enhancement
    kern_hpgb = kernel size for hpgb 
    sigma = number of sigmas for hpgb
    rb_radius = rolling ball size
    hyb_offset = number of hybcycles to subtract for file name adjustment
    p_min = minimum percentile
    p_max = maximum percentile
    norm_int = bool to normalize intensity
    rollingball = do a rolling ball subtraction
    lowpass = do a low pass gaussian filter
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    tophat_raw = bool to perform tophat on raw image
    """
    #output path
    parent = Path(image_path).parent
    while "seqFISH_datapipeline" not in os.listdir(parent):
        parent = parent.parent
    output_dir = parent / "seqFISH_datapipeline" /"output"/ "pre_processed_images"
    hybcycle =  Path(image_path).parent.name
    output_path = output_dir / hybcycle / Path(image_path).name
    
    if hyb_offset != 0:
        hyb_number = int(output_path.parent.name.split("_")[1])
        new_number = "HybCycle_" + str(hyb_number-hyb_offset)
        pos_name = Path(image_path).name
        output_path = output_path.parent.parent / new_number /pos_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read images
    image = pil_imread(image_path)
    if image.shape[1] != num_channels:
        image = np.swapaxes(image, 0, 1)
    if type(stack_bkgrd) != type(None):
        bkgrd = pil_imread(stack_bkgrd, swapaxes=True)
        if bkgrd.shape[1] != num_channels:
            bkgrd = pil_imread(stack_bkgrd, swapaxes=False)
    
    #background correct
    if type(stack_bkgrd) != type(None):
        corr_img = background_correct_image(image, correction_type, bkgrd, 
                                            gamma, kern_hpgb, sigma, match_hist, 
                                            subtract, divide, tophat_raw)
    else:
        corr_img = background_correct_image(image, correction_type, stack_bkgrd,
                                            gamma, kern_hpgb, sigma, 
                                            match_hist, subtract, divide, tophat_raw)
            
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
        if norm_int == True:
            lpgb = scale_int(lpgb, p_min=p_min, p_max=p_max)
        tf.imwrite(str(output_path), lpgb)
    else:
        if norm_int == True:
            corr_img = scale_int(corr_img, p_min=p_min, p_max=p_max)
        print('writing image')
        tf.imwrite(str(output_path), corr_img)

def correct_many(images, correction_type = None, stack_bkgrd=None, num_channels=4,
                 gamma = 1.4,kern_hpgb=5, sigma=40, rb_radius=5, hyb_offset=0,p_min=80,
                 p_max = 99.999, norm_int = True,
                 rollingball=False, lowpass = True, match_hist=True, subtract=True, divide=False, tophat_raw=False):
    """
    function to correct all image
    
    Parameters
    ----------
    image_path= path to image
    correction_type = which correction algo to use
    stack_bkgrd = 4 or 3d array of background image
    num_channels = number of expected channels for axes checking
    gamma = int for gamma enhancement
    kern_hpgb = kernel size for hpgb 
    sigma = number of sigmas for hpgb
    rb_radius = rolling ball size
    hyb_offset = number of hybcycles to subtract for file name adjustment
    p_min = minimum percentile
    p_max = maximum percentile
    norm_int = bool to normalize intensity
    rollingball = do a rolling ball subtraction
    lowpass = do a low pass gaussian filter
    match_hist = bool to match histograms of blurred image
    subtract = bool to subtract blurred image from raw
    divide = bool to divide blurred image from raw
    tophat_raw = bool to perform tophat on raw image
    """
    import time
    start = time.time()
    
    if type(images) != list:
        bkgrd_corr_one(images, correction_type,stack_bkgrd, num_channels,  
                       gamma, kern_hpgb,sigma, rb_radius, hyb_offset, p_min,
                       p_max, norm_int, rollingball, 
                       lowpass,  match_hist, subtract, divide, tophat_raw)
        
        
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(bkgrd_corr_one, path, correction_type, stack_bkgrd,
                                 num_channels, gamma,kern_hpgb, sigma, rb_radius,hyb_offset,
                                 p_min, p_max, norm_int,
                                 rollingball, lowpass,  match_hist, subtract, divide, tophat_raw)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
