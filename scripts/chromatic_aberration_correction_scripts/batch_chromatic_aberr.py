from chromatic_aberration_correction import chromatic_corr_offsets, apply_chromatic_corr
from pathlib import Path
import os
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#get all hybs for specific pos
directory = Path("/groups/CaiLab/personal/Lex/raw/230623_4k_0.5nM_IP/pyfish_tools/output/z_matched_images/fiducial_aligned")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#reference image
ref_img = f"/groups/CaiLab/personal/Lex/raw/230623_4k_0.5nM_IP/chromatic_aberration/{position_name}"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_img, threshold_abs=900,  max_dist=1.25, ransac_threshold = 0.5, num_channels=4)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12,  num_channels=4, write = True)
