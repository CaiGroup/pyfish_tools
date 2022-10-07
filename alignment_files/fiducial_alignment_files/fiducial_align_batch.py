#custom function
from fiducial_alignment_affine import fiducial_align_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/notebook_pyfiles/dapi_aligned/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#use this for all hyb alignment
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#directory to beads
ref = f"/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/chromatic_aberration/{position_name}"

tiff_list = files #list of images
ref_src = ref #reference bead images
threshold_abs=500 #raw intensity value the dots must be over
max_dist=1 #maximum allowed distance a fiducial can be prior to alignment
ransac_threshold=0.5 #maximum pixel distance a dot has to be after correction to be considered an inlier
bead_channel_single=None #if all channels have beads set to None, else specificy which channel (0,1,2,3)
include_dapi=False #bool to include dapi channel
use_ref_coord = True # use the reference coordinates to find moving dots 
swapaxes=True #swap z and c axis
cores = 16 #number of cores to use

fiducial_align_parallel(tiff_list, ref_src, threshold_abs=threshold_abs, max_dist=max_dist,
                        ransac_threshold=ransac_threshold,bead_channel_single=bead_channel_single,
                        include_dapi=include_dapi,use_ref_coord=use_ref_coord, swapaxes=swapaxes, cores=cores)
