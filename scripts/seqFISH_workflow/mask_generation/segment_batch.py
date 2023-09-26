#import functions
from cellpose_segmentation import *
import nuclear_cyto_match as ncm
import glob
import re
import tifffile as tf
from pathlib import Path
import os
import sys
from util import find_matching_files

# User defined settings
#----------------------------------------------------------------------------------
input_directory = "/groups/CaiLab/personal/Lex/raw/230726_43gene_smfish/segmentation/*.tif" # don't forget the star
num_pos         = 56
max_project     = False
have_multiple_z = True
channel         = 0   #which channel has segmenation marker (0,1,2,3)?
diameter_cyto   = 350 #diameter in pixels for cytoplasm
diameter_nucl   = 150 #diameter in pixels for nucleus
flow            = 2
cellprob        = -1
num_channels    = 3 
save_dir_cyto   = '/groups/CaiLab/personal/Lex/raw/230726_43gene_smfish/pyfish_tools/output/masks/cyto' 
save_dir_nucl   = '/groups/CaiLab/personal/Lex/raw/230726_43gene_smfish/pyfish_tools/output/masks/nucl'
repeat          = 1    #number of copies for z's if you wish to propagate (1-infinite)
threshold       = 0.20 #percent coverage of nuclear mask on top of cytoplasm mask
use_gpu         = False #use this at your own risk. Job submission can take forever if GPUs are requested.
#----------------------------------------------------------------------------------
# NO NEED TO EDIT
if use_gpu == True:
    core.use_gpu(gpu_number=1, use_torch=True)
    model_cyto      = models.Cellpose(gpu=True, model_type="cyto2")
    model_nucl      = models.Cellpose(gpu=True, model_type="nuclei")
else:
    model_cyto      = models.Cellpose(gpu=False, model_type="cyto2")
    model_nucl      = models.Cellpose(gpu=False, model_type="nuclei")

files           = glob.glob(input_directory)
key             = [int(re.search('MMStack_Pos(\\d+)', f).group(1)) for f in files]
files           = list(np.array(files)[np.argsort(key)])
imgs            = read_images(files[:num_pos], num_channels=num_channels, max_project=max_project)
imgs_final      = generate_final_images(imgs, have_multiple_z=have_multiple_z, channel=channel)

channels        = [0,0]
masks, _, _, _  = model_cyto.eval(imgs_final, diameter=diameter_cyto, channels=channels, 
                             flow_threshold=flow, cellprob_threshold=cellprob, do_3D=False)
write_masks(masks, files, save_dir_cyto, repeat_mask_multi_z = repeat)

imgs_final      = generate_final_images(imgs, have_multiple_z=have_multiple_z, channel=-1)
channels        = [0,0]
masks, _, _, _  = model_nucl.eval(imgs_final, diameter=diameter_nucl, channels=channels, 
                             flow_threshold=flow, cellprob_threshold=cellprob, do_3D=False)
write_masks(masks, files, save_dir_nucl, repeat_mask_multi_z = repeat)

#read in masks
if save_dir_cyto[-1] != "/":
    save_dir_cyto = save_dir_cyto + "/"
if save_dir_nucl[-1] != "/":
    save_dir_nucl = save_dir_nucl + "/"
nuc_paths  = glob.glob(f"{save_dir_nucl}*")
cyto_paths = glob.glob(f"{save_dir_cyto}*")

#organize files numerically
key        = [int(re.search('MMStack_Pos(\\d+)', f).group(1)) for f in nuc_paths]
nuc_paths  = list(np.array(nuc_paths)[np.argsort(key)])

key        = [int(re.search('MMStack_Pos(\\d+)', f).group(1)) for f in cyto_paths]
cyto_paths = list(np.array(cyto_paths)[np.argsort(key)])

#match masks
nuclear = []
cyto    = []
for i in tqdm(range(len(nuc_paths))):
    nuclear.append(pil_imread(nuc_paths[i]))
    cyto.append(pil_imread(cyto_paths[i]))
#match nuclear and cyto masks
cyto_new = ncm.nuclear_cyto_matching(cyto, nuclear, threshold = threshold)

#set output paths
parent = Path(nuc_paths[0]).parent
while "pyfish_tools" not in os.listdir(parent):
    parent = parent.parent
output_folder = parent / "pyfish_tools"/ "output" / "final_masks"
output_folder.mkdir(parents=True, exist_ok=True)
#write images
for i, lab_img in enumerate(cyto_new):
    imagename = Path(cyto_paths[i]).name
    path = output_folder / imagename
    tf.imwrite(str(path), lab_img)