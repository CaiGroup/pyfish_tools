"""
author: Katsuya Lex Colon & Arun Chakravorty (SIFT portion)
updated: 02/13/24
"""

from skimage import registration
from scipy import ndimage
import tifffile as tf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from helpers.util import pil_imread
from skimage.metrics import structural_similarity
import cv2


def filter_keypoints_by_intensity(keypoints, descriptors, image, intensity_threshold=7):
    filtered_keypoints = []
    filtered_descriptors = []
    for i, keypoint in enumerate(keypoints):
        if image[int(keypoint.pt[1]), int(keypoint.pt[0])] >= intensity_threshold:
            filtered_keypoints.append(keypoint)
            filtered_descriptors.append(descriptors[i])
    return filtered_keypoints, np.array(filtered_descriptors)


def dapi_alignment_single_BF_Single_WithBlur_changed(ref, moving, num_channels):
    """A function to obtain translational offsets using phase correlation. Image input should have the format z,c,x,y.
    Parameters
    ----------
    ref: Hyb 0 image path
    moving: image you are trying to align path
    
    Output
    -------
    image (c,z,x,y)
    """

    #create output path
    parent = Path(moving).parent
    while "pyfish_tools" not in os.listdir(parent):
        parent = parent.parent
    output_folder = parent / "pyfish_tools" / "output"/ 'dapi_aligned'
    hybcycle = Path(moving).parent.name
    output_path = output_folder / hybcycle / Path(moving).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        image_ref = pil_imread(ref, num_channels = None, swapaxes = True)
        if image_ref.shape[1] != num_channels:
            image_ref = pil_imread(ref, num_channels = None, swapaxes = False)
            if image_ref.shape[0] == image_ref.shape[1]:
                image_ref = check_axis(image_ref)
            if image_ref.shape[1] != num_channels:
                raise Exception("Error reading image file, will try to read it another way")
    except:
        image_ref = pil_imread(ref, num_channels = num_channels, swapaxes = True)
        if image_ref.shape[1] != num_channels:
            image_ref = pil_imread(ref, num_channels = num_channels, swapaxes = False)
            if image_ref.shape[0] == image_ref.shape[1]:
                image_ref = check_axis(image_ref)
                
    try:
        image_moving = pil_imread(moving, num_channels = None, swapaxes = True)
        if image_moving.shape[1] != num_channels:
            image_moving = pil_imread(moving, num_channels = None, swapaxes = False)
            if image_moving.shape[0] == image_moving.shape[1]:
                image_moving = check_axis(image_moving)
            if image_moving.shape[1] != num_channels:
                raise Exception("Error reading image file, will try to read it another way")
                
    except:
        image_moving = pil_imread(moving, num_channels = num_channels, swapaxes = True)
        if image_moving.shape[1] != num_channels:
            image_moving = pil_imread(moving, num_channels = num_channels, swapaxes = False)
            if image_moving.shape[0] == image_moving.shape[1]:
                image_moving = check_axis(image_moving)
    
   # get dapi channel for reference and moving assuming it is at the end
    dapi_ref = image_ref.shape[1]-1
    dapi_moving = image_moving.shape[1]-1
    
    # max project dapi channel
    max_proj_ref = np.max(np.swapaxes(image_ref,0,1)[dapi_ref], axis=0)
    max_proj_moving = np.max(np.swapaxes(image_moving,0,1)[dapi_moving], axis=0)
    
    # Apply Gaussian blur to the max projected images
    #max_proj_ref = cv2.GaussianBlur(max_proj_ref, (7, 7), 0)
    #max_proj_moving = cv2.GaussianBlur(max_proj_moving, (7, 7), 0)

    # Apply median filter to the max projected images
    #max_proj_ref = cv2.medianBlur(max_proj_ref.astype(np.float32), 5)
    #max_proj_moving = cv2.medianBlur(max_proj_moving.astype(np.float32), 5)

    #Apply Bilateral filter
    max_proj_ref = cv2.bilateralFilter(max_proj_ref.astype(np.float32), d=20, sigmaColor=100, sigmaSpace=100)
    max_proj_moving = cv2.bilateralFilter(max_proj_moving.astype(np.float32), d=20, sigmaColor=100, sigmaSpace=100)

    
    # Scale the intensity values
    max_intensity = 10000
    max_proj_ref[max_proj_ref > max_intensity] = max_intensity
    max_proj_moving[max_proj_moving > max_intensity] = max_intensity
    max_proj_ref = cv2.normalize(max_proj_ref, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    max_proj_moving = cv2.normalize(max_proj_moving, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print("Finishing Filtering")
    
    sift_broad = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=5)
    keypoints_ref, descriptors_ref = sift_broad.detectAndCompute(max_proj_ref, None)
    keypoints_moving, descriptors_moving = sift_broad.detectAndCompute(max_proj_moving, None)
    
    #keypoints_ref, descriptors_ref = filter_keypoints_by_intensity(keypoints_ref, descriptors_ref, max_proj_ref)
    #keypoints_moving, descriptors_moving = filter_keypoints_by_intensity(keypoints_moving, descriptors_moving, max_proj_moving)
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_moving)

    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_moving[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    ### Use RANSAC to filter out outliers and retain matches that have a consistent transformation model
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = mask.ravel().tolist()
    good_matches = [m for m, inlier in zip(matches, inliers) if inlier]
    
    #matches = sorted(matches, key=lambda x: x.distance)
    #good_matches = matches #[:int(len(matches) * 0.8)]  # Use only the top 50% matches

    print("Number of Good Matches", len(good_matches))    

    points_ref = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_moving = np.float32([keypoints_moving[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.estimateAffinePartial2D(points_moving, points_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if matrix is not None:
        print("MATRIX IS NOT NONE")
        dx = matrix[0, 2]
        dy = matrix[1, 2]

        print(dy, dx)
        
        try:
            aligned_moving = cv2.warpAffine(max_proj_moving, matrix, (max_proj_ref.shape[1], max_proj_ref.shape[0]))
            mse = np.mean((max_proj_ref - aligned_moving)**2)
            ssim = structural_similarity(max_proj_ref, aligned_moving)
            print('FINSHING UP HERE')
            print(f"Alignment accuracy: MSE = {mse:.4f}, SSIM = {ssim:.4f}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
    else:
        print("Failed to estimate transformation matrix.")
        dx, dy = 0, 0

    shift = [dy, dx]
    print("Calculated shift", shift)

    
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
    print('OUTPUTTING FILE', str(output_path))
    tf.imwrite(str(output_path), corr_stack)
    #write shift
    pos = output_path.name.split("_")[1].replace(".ome.tif","_shift.txt")
    shift_output = output_path.parent/pos
    np.savetxt(str(shift_output),shift)
    del corr_stack

def dapi_alignment_parallel(image_ref, images_moving, num_channels):
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
            #dapi_alignment_single_BF_Single_WithBlur
            #exe.submit(dapi_alignment_single, image_ref, images_moving, num_channels)
            exe.submit(dapi_alignment_single_BF_Single_WithBlur_changed, image_ref, images_moving, num_channels)
    
    else:
        with ThreadPoolExecutor(max_workers=20) as exe:
            futures = {}
            for path in images_moving:
                #fut = exe.submit(dapi_alignment_single, image_ref, path, num_channels)
                fut = exe.submit(dapi_alignment_single_BF_Single_WithBlur_changed, image_ref, path, num_channels)
                futures[fut] = path
        
            for fut in as_completed(futures):
                path = futures[fut]
                print(f'Path {path} completed after {time.time() - start} seconds')
   
