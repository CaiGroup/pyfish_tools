#custom function
from fiducial_alignment_affine import fiducial_align_parallel
from pathlib import Path
import os
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from helpers.util import find_matching_files

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#general path and position name
JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/pyfish_tools/output/z_matched_images")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#use this for all hyb alignment
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#directory to beads
ref = f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/pyfish_tools/output/z_matched_images/beads/{position_name}"

tiff_list           = files #list of images
ref_src             = ref #reference bead images
threshold_abs       = 800 #raw intensity value the dots must be over
max_dist            = 1 #maximum allowed distance a fiducial can be prior to alignment. Note: Set pixel distance to 5 is you are aligning DAPI punctates.
ransac_threshold    = 0.20 #maximum pixel distance a dot has to be after correction to be considered an inlier
bead_channel_single = None #if all channels have beads set to None, else specificy which channel (0,1,2,3). Note: You can try to use DAPI for affine alignment.
include_dapi        = False #Set to True if you are using DAPI punctates (euchromatin) for affine
use_ref_coord       = True # use the reference coordinates to find moving dots 
num_channels        = 4 #number of channels in image
cores               = 16 #number of cores to use
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#no need to edit
fiducial_align_parallel(tiff_list, ref_src, threshold_abs=threshold_abs, max_dist=max_dist,
                        ransac_threshold=ransac_threshold,bead_channel_single=bead_channel_single,
                        include_dapi=include_dapi,use_ref_coord=use_ref_coord, num_channels=num_channels, cores=cores)