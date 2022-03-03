from chromatic_aberration_correction import *
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/020822_erna_dash_tf/notebook_pyfiles/dapi_aligned/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#hybcycle images
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#reference image
ref_img = f"/groups/CaiLab/personal/Lex/raw/020822_erna_dash_tf/chromatic_aberration/{position_name}"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_img, region_size=9, min_distance=10, 
                          threshold_abs=500, num_peaks=500, max_dist=5,ransac_threshold = 0.5,
                          use_488 = False, swapaxes=True)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12, use_488=False, swapaxes=True, write = True)