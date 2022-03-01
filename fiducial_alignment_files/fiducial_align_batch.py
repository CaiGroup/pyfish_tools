#custom function
from fiducial_alignment_affine import fiducial_align_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#use this for all hyb alignment
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#directory to beads
ref = f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/beads/{position_name}"

tiff_list = files #list of images
ref_src = ref #reference bead images
region_size=7 #bounding box for gaussian fitting
min_distance=10 #distance between peaks for initial dot finding
threshold_abs=1000 #raw intensity value the dots must be over
num_peaks=1000 #number of dots to use
max_dist=2 #maximum allowed distance a fiducial can be prior to alignment
ransac_threshold=0.5 #maximum pixel distance a dot has to be after correction to be considered an inlier
include_dapi=False #bool to include dapi channel
swapaxes=True #swap z and c axis
cores = 32 #number of cores to use

fiducial_align_parallel(tiff_list, ref_src, region_size=region_size, min_distance=min_distance, 
                        threshold_abs=threshold_abs, num_peaks=num_peaks, max_dist=max_dist,
                        ransac_threshold=ransac_threshold,
                        include_dapi=include_dapi, swapaxes=swapaxes, cores=cores)
