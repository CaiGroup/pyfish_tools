from chromatic_aberration_correction import chromatic_corr_offsets, apply_chromatic_corr
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
_, _, tform = chromatic_corr_offsets(ref_img, threshold_abs=500,  max_dist=3, ransac_threshold = 0.5, swapaxes=True)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12,  swapaxes=True, write = True)
