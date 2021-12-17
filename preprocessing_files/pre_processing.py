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

def background_correct_image(stack, correction_algo, stack_bkgrd=None, z=2, size=2048, gamma = 1.4, sigma=40):
    '''
    INPUT:  - stack: a TIF stack numpy array in the format (Z-axis position, Channel, Y-axis position, X-axis position),
                     with elements as 16-bit unsigned numpy integers
                     
                     Note: The last channel is assumed to be the nuclear stain. Only dimming correction via LS regression
                     is applied to that channel.
                     
            - stack_bkgrd: aligned initial background image
            
            - correction_algo: the name of one of the image 
                               processing algorithms below (SigmaClipping_and_Gamma_C,
                               Gaussian_and_Gamma_Correction,LSR_Backgound_Corrrection)
            - z: number of z's
            
            -gamma: int
            -sigma:int
            
    OUTPUT: - a TIF stack numpy array in the format (Z-axis position, Channel, Y-axis position, X-axis position),
              with elements as 16-bit unsigned numpy integers 
    
    '''
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
            channel = z_slice[i]
            channel = np.asarray([float(i) for i in channel.flatten()]).reshape(channel.shape)
            if correction_algo != LSR_Backgound_Correction:
                if i == (z_slice.shape[0]-1):
                    correction_algo = LSR_Backgound_Correction
                    corrected_channel = correction_algo(channel)
                else:
                    if correction_algo == Gaussian_and_Gamma_Correction:
                        corrected_channel = correction_algo(channel, gamma, sigma)
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

def tophat_background(image, kern=3):
    """tophat raw then initial background subtraction"""
    # Getting the kernel to be used in Top-Hat
    filterSize =(kern, kern)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    # Applying the Top-Hat operation
    tophat_img = cv2.morphologyEx(image,cv2.MORPH_TOPHAT, kernel)
    
    return tophat_img

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

def Gaussian_and_Gamma_Correction(image, gamma, sigma):
    """ Background Correction via Background Elimination (estimated by Gaussian), followed by Gamma Correction"""
    image_blur = ndimage.gaussian_filter(image, sigma)
    image_background_eliminated= image/((image_blur)/np.mean(image_blur))
    contrast_ench_Gamma_c = adjust_gamma(image_background_eliminated, gamma)
    contrast_ench_Gamma_c = (contrast_ench_Gamma_c/np.mean(contrast_ench_Gamma_c))*np.mean(image)
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
    
def high_pass_gaussian(img, kern=9, sigma=1):
    """A high pass gaussian filter
    Parameters
    ----------
    img = z,c,x,y
    kern = int
    """
    if len(img.shape) == 3:
        #generate kernel
        kernel = np.ones((kern,kern),np.float32)/kern**2
        #blur the image and subtract
        c_slice = []
        for c in range(img.shape[0]):
            #gaussian filter
            blur = cv2.GaussianBlur(img[c],(kern,kern),sigma)
            #subtract
            filtered = util.img_as_int(img[c])-util.img_as_int(blur)
            #set negative values to zero
            filtered[filtered<0]=0
            c_slice.append(filtered)
        return np.array(c_slice) 
    else:
        #generate kernel
        kernel = np.ones((kern,kern),np.float32)/kern**2
        #blur the image and subtract
        z_slice = []
        for z in range(img.shape[0]):
            channel_slice = []
            for c in range(img.shape[1]):
                #gaussian filter
                blur = cv2.GaussianBlur(img[z][c],(kern,kern),sigma)
                #subtract
                filtered = util.img_as_int(img[z][c])-util.img_as_int(blur)
                #set negative values to zero
                filtered[filtered<0]=0
                channel_slice.append(filtered)
            z_slice.append(channel_slice)
        return np.array(z_slice)



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
    """Assuming a gaussian psf, images are deconvoluted using the richardson-lucy algorithm
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

def deconvolute_one(image_path, image_ref, kern_hpgb = 9, sigma_hpgb = 1, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    size=9,min_distance=10,threshold_abs=1000,
                    num_peaks=1000, edge='raise', swapaxes=True,
                    noise= True, bkgrd_corr = True, bkgrd_sub=True, remove_fiducial=False):
    
    """deconvolute one image only
    Parameters
    ----------
    image_path = path to single image
    image_ref = path to reference image for removing fiducials
    kern_hpgb = kernel size for high-pass gaussian blur
    sigma_hpgb = sigma for high-pass gaussian blur
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
    edge = argument for bounding box in remove fiducials
    swapaxes = bool to swap axes when readin in an image
    noise = re-convolve image at the end
    bkgrd_corr = use background correction algo before deconvolution
    bkgrd_sub = bool to perform background subtraction
    remove_fiducial = bool to keep or remove fiducials
    """
    
    #make output directory
    orig_image_dir = Path(image_path).parent.parent
    output_folder = Path(orig_image_dir).with_name('deconvoluted_images')
    output_path = output_folder / Path(image_path).relative_to(orig_image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    #read in image
    if swapaxes == True:
        image = tf.imread(image_path)
        image = np.swapaxes(image,0,1)
        if (bkgrd_corr == True) and (bkgrd_sub==True):
            bkgrd_tiff_src =  Path(image_path).parent.parent / "final_background"
            pos = Path(image_path).name
            stack_bkgrd_path = bkgrd_tiff_src / pos
            stack_bkgrd = tf.imread(stack_bkgrd_path)
            stack_bkgrd = np.swapaxes(stack_bkgrd,0,1)
        else:
            stack_bkgrd=None
    else:
        image = tf.imread(image_path)
        if (bkgrd_corr == True) and (bkgrd_sub==True):
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
    if bkgrd_corr == True:
        if len(image.shape) ==4:
            z=image.shape[0]
            size = image.shape[2]
            print('background correction...')
            hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, stack_bkgrd, z, size) 
        else:
            z=1
            size = image.shape[2]
            print('background correction...')
            hpgb_image = background_correct_image(image,Gaussian_and_Gamma_Correction, stack_bkgrd, z, size) 
    else:
        print('high pass gaussian...')
        hpgb_image = high_pass_gaussian(image, kern=kern_hpgb, sigma = sigma_hpgb)
        
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
            
def deconvolute_many(images, image_ref, kern_hpgb = 9, sigma_hpgb = 1, kern_rl = 5, 
                    kern_lpgb = 3, sigma=(1.8,1.6,1.5,1.3), radius=(4,4,4,4),
                    model="gaussian", microscope="boc",
                    size=9,min_distance=10,threshold_abs=1000,
                    num_peaks=1000, edge='raise', swapaxes=True,
                    noise= True, bkgrd_corr = True, bkgrd_sub=True, remove_fiducial=False):
    
    """function to deconvolute all images
     Parameters
    ----------
    images = list of image paths
    image_ref = path to reference image for removing fiducials
    kern_hpgb = kernel size for high-pass gaussian blur
    sigma_hpgb = sigma for high-pass gaussian blur
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
    edge = argument for bounding box in remove fiducials
    swapaxes = bool to swap axes when readin in an image
    noise = re-convolve image at the end
    bkgrd_corr = use background correction algo before deconvolution
    remove_fiducial = bool to keep or remove fiducials
    """
    
    import time
    start = time.time()
    
    if type(images) != list:
        deconvolute_one(images, image_ref, kern_hpgb=kern_hpgb, 
                       sigma_hpgb=sigma_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       size=size,min_distance=min_distance,threshold_abs=threshold_abs,
                       num_peaks=num_peaks, edge=edge, swapaxes=swapaxes,
                       noise=noise, bkgrd_corr=bkgrd_corr, bkgrd_sub=bkgrd_sub, remove_fiducial=remove_fiducial)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(deconvolute_one, path, image_ref, kern_hpgb=kern_hpgb, 
                       sigma_hpgb=sigma_hpgb, kern_rl=kern_rl, 
                       kern_lpgb=kern_lpgb, sigma=sigma, radius=radius,model=model, microscope=microscope,
                       size=size,min_distance=min_distance,threshold_abs=threshold_abs,
                       num_peaks=num_peaks, edge=edge, swapaxes=swapaxes,
                       noise=noise, bkgrd_corr=bkgrd_corr, bkgrd_sub=bkgrd_sub, remove_fiducial=remove_fiducial)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
                
def bkgrd_corr_one(image_path, correction_type = None, stack_bkgrd=None, swapaxes=False, 
                   z=2, size=2048, gamma = 1.4, sigma = 40, rb_radius=5, rollingball = False, lowpass=True):
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
    gamma = int
    simga = int
    lowpass = do a low pass gaussian filter
    rollingball = do a rolling ball subtraction
    """
    
    orig_image_dir = Path(image_path).parent.parent
    output_folder = Path(orig_image_dir).with_name('pre_processed_images')
    output_path = output_folder / Path(image_path).relative_to(orig_image_dir)
    
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
        corr_img = background_correct_image(image, correction_type, bkgrd, z, size, gamma, sigma)
    else:
        corr_img = background_correct_image(image, correction_type, stack_bkgrd, z, size, gamma, sigma)
    
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
        tf.imwrite(str(output_path).replace("ome.tif","tif"), lpgb)
    else:
        print('writing image')
        tf.imwrite(str(output_path).replace("ome.tif","tif"), corr_img)

def correct_many(images, correction_type = None, stack_bkgrd=None, swapaxes=False,
                 z=2, size=2048, gamma = 1.4, sigma=40, rb_radius=5, rollingball=False, lowpass = True):
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
    """
    import time
    start = time.time()
    
    if type(images) != list:
        bkgrd_corr_one(images, correction_type,stack_bkgrd, swapaxes, z, size,  
                       gamma, sigma, rb_radius, rollingball, lowpass)
    else:
        with ProcessPoolExecutor(max_workers=12) as exe:
            futures = {}
            for path in images:
                fut = exe.submit(bkgrd_corr_one, path, correction_type, stack_bkgrd,
                                 swapaxes, z, size,  gamma, sigma,rb_radius,
                                 rollingball, lowpass)
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')