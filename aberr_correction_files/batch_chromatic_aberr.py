from chromatic_aberration_correction import *
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/dapi_aligned/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#hybcycle images
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#reference image
ref_directory = "/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/chromatic_aberration_lb/new_probes_5/new_probes_5_MMStack_Pos0.ome.tif"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_directory,region_size=7, min_distance=10, 
                          threshold_abs=500, num_peaks=500, max_dist=5,
                          include_dapi=False, swapaxes=True)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12, include_dapi=False, swapaxes=True, write = True)