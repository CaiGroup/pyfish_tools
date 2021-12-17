#custom function
from fiducial_alignment_affine import fiducial_align_parallel
#matlab engine for radial centering
import matlab.engine
#file organization packages
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

# ##get all hybs with same position
# ##use this for background alignment
# files = directory / position_name

#use this for all hyb alignment
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

# #get hyb 0 with same position
# ##use this for background alignment
# directory_align=Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/beads")
# ref, _, _ = find_matching_files(directory_align, 'HybCycle_0' + f'/{position_name}') 

# ##use this for normal alignment
# #ref, _, _ = find_matching_files(directory, 'HybCycle_0' + f'/{position_name}') 
ref = f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/beads/{position_name}"

tiff_list = files
ref_src = ref
region_size=7
min_distance=10
threshold_abs=500
num_peaks=500
max_dist=2
eng = matlab.engine.start_matlab()
radial_center=True
include_dapi=True
swapaxes=True
cores = 12

fiducial_align_parallel(tiff_list, ref_src, region_size=region_size, min_distance=min_distance, 
                       threshold_abs=threshold_abs, num_peaks=num_peaks, max_dist=max_dist,eng=eng,
                        radial_center=radial_center,include_dapi=include_dapi, swapaxes=swapaxes, cores=cores)
