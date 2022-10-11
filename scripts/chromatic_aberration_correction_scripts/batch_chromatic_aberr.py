from chromatic_aberration_correction import chromatic_corr_offsets, apply_chromatic_corr
from pathlib import Path
import os
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#get all hybs for specific pos
directory = Path("/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/seqFISH_datapipeline/output/z_matched_images")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

# #this is for unique file names
# files = ["/groups/CaiLab/personal/Lex/raw/Ratio_metric_repeat/notebook_pyfiles/max_projected/EEF2_ratiometric_Rlvl6_Llvl2_premixed_730,640,561,488,405_2000,200,300,400,400_Pos2.ome.tif", "/groups/CaiLab/personal/Lex/raw/Ratio_metric_repeat/notebook_pyfiles/max_projected/EEF2_ratiometric_Rlvl6_Llvl3_premixed_730,640,561,488,405_2000,200,300,400,400_Pos2.ome.tif"]

#reference image
ref_img = f"/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/chromatic_aberration/MMStack_Pos{JOB_ID}.ome.tif"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_img, threshold_abs=500,  max_dist=3, ransac_threshold = 0.5, num_channels=4)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12,  num_channels=4, write = True)
