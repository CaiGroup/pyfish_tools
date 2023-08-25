from chromatic_aberration_correction import chromatic_corr_offsets, apply_chromatic_corr
from pathlib import Path
import os
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#get all hybs for specific pos
directory = Path("/groups/CaiLab/personal/Lex/raw/230708_3k_1nM_split_IP_tmg/pyfish_tools/output/dapi_aligned/fiducial_aligned")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#reference image
ref_img = f"/groups/CaiLab/personal/Lex/raw/230708_3k_1nM_split_IP_tmg/chromatic_aberration/{position_name}"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_img, threshold_abs=950,  max_dist=1.5, ransac_threshold = 0.4, num_channels=4)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12,  num_channels=4, write = True)