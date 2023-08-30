"""
author: Katsuya Lex Colon
date: 08/29/23
"""

#data processing packages
import numpy       as     np
import re
import cv2
import tifffile    as     tf
import pandas      as     pd
from   collections import defaultdict, Counter
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from   helpers.util import pil_imread
#file management
from   glob         import glob
import os
from   pathlib      import Path

def z_matching(image_dir, pos_number = 0):
    
    """
    Function to find matching z across hybs for a given pos.
    
    Parameters
    ----------
    image_dir: directory of where your images are located
    pos_number: the position you wish to align
    """
    
    #output directory
    parent = Path(image_dir).parent
    while "pyfish_tools" not in os.listdir(parent):
        parent = parent.parent
    
    output_dir = parent / "pyfish_tools" / "output"/ "z_matched_images"
    
    #adjust string path if missing "/"
    if image_dir[-1] != "/":
        image_dir = image_dir + "/"
        
    #get all hybcycles
    hyb_images = glob(image_dir + f"HybCycle_*/MMStack_Pos{pos_number}.ome.tif")
    #organize hybcycles numerically
    key = [int(re.search('HybCycle_(\\d+)', f).group(1)) for f in hyb_images]
    files = list(np.array(hyb_images)[np.argsort(key)])
    
    #get ref image (which is hyb0)
    ref_path = files[0]
    
    print(f"reference image is: {ref_path}")
    
    #remove first line in files list
    del files[0]
    
    #collect matching z info by performing normalized cross correlation 
    match_z = []
    z_mapper = []
    #read ref image
    ref = pil_imread(ref_path, swapaxes=False)
    for file in files:
        hyb_list = []
        #read in moving z images
        src = pil_imread(file, swapaxes=False)
        #get dapi channel
        dapi_ch = ref.shape[1]-1
        for z in range(ref.shape[0]):
            #collect correlation info
            ccoef_list = []
            ref_compressed = ref[z][dapi_ch].astype(np.float32)
            #compress images to 32 float for template matching
            for z_2 in range(ref.shape[0]):
                src_compressed = src[z_2][dapi_ch].astype(np.float32)
                corr = cv2.matchTemplate(ref_compressed, src_compressed, cv2.TM_CCOEFF_NORMED)
                ccoef_list.append(corr)
            #find best z
            best_z = np.argmax(ccoef_list)
            #store ref z, best matching z to ref, and r
            hyb_list.append([z, best_z, ccoef_list[best_z][0][0]])
        #if there are any colliding z, pick best one
        grouped_data = defaultdict(list)
        for entry in hyb_list:
            key = entry[1]  # Use the 2nd axis value as the key
            grouped_data[key].append(entry)
        curated_hyb_list = []
        for key in grouped_data:
            best = np.argmax(np.array(grouped_data[key])[:,2])
            curated_hyb_list.append(grouped_data[key][best])
        match_z.append(curated_hyb_list)
        #create table reference
        z_mapped = {}
        for z1,z2,_ in curated_hyb_list:
            z_mapped.update({z1:z2})
        z_mapper.append(z_mapped)
        
    #check what the maximum allowed z stack can be
    #by looking for minimum number of z across all hybcycles
    num_of_zs = []
    for fov in match_z:
        num_of_zs.append(len(fov))
    max_z_allowed = min(num_of_zs)
    
    #use lowest z slice that appears the most for starting ref
    obj = Counter(np.vstack(match_z)[:,0])
    df = pd.DataFrame.from_dict(obj, orient="index")
    df = df.reset_index().sort_values("index")
    ref_start = int(df["index"][df[0].argmax()])
    
    #now offset the zs for reference
    if max_z_allowed == 1:
        ref = ref[ref_start,:,:,:]
        ref_zs = [ref_start]
    else:
        ref = ref[ref_start:ref_start+(max_z_allowed-1), :, :, :]
        ref_zs = np.arange(ref_start,ref_start+(max_z_allowed), 1).astype(int)

    hyb_folder = Path(ref_path).parent.name
    output_path = output_dir / hyb_folder
    output_path.mkdir(parents=True, exist_ok=True)
    if len(ref) != 0:
        tf.imwrite(str(output_path / f"MMStack_Pos{pos_number}.ome.tif" ), ref)

    #now offset zs for rest of hybs
    for i in range(len(match_z)):
        #moving zs
        src = pil_imread(files[i], swapaxes=False)
        #hybcycle name
        hyb_folder = Path(files[i]).parent.name
        #check if file already exists
        if os.path.isfile(str(output_dir/f"pos{pos_number}_matched_z_info.txt")):
            log = open(str(output_dir/f"pos{pos_number}_matched_z_info.txt"), "a")
        else:
            log = open(str(output_dir/f"pos{pos_number}_matched_z_info.txt"), "w+")
        #get correct z mapper
        table_map = z_mapper[i]
        #only get matching z from reference 
        offset_image = []
        for ref_z in ref_zs:
            try:
                offset_image.append(src[table_map[ref_z],:,:,:])
                #record ref z and src z
                log.write(f"{hyb_folder}: ref_z = {ref_z}, src_z = {table_map[ref_z]}\n")
            except:
                #if z is really off, then there will be an error here
                #instead it will output unaligned image
                #this will be recoded in log file
                log.write(f"{hyb_folder}: This FOV has very large z fluctuations.\n")
                break
        if offset_image != []: 
            offset_image = np.array(offset_image)    
        else:
            offset_image = src
        output_path = output_dir / hyb_folder
        output_path.mkdir(parents=True, exist_ok=True)
        tf.imwrite(str(output_path / f"MMStack_Pos{pos_number}.ome.tif" ), offset_image)
        #close log file
        log.close()
    
    