from chromatic_aberration_correction import chromatic_corr_offsets, apply_chromatic_corr
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

files = ["/groups/CaiLab/personal/Lex/raw/Ratio_metric_repeat/notebook_pyfiles/max_projected/EEF2_ratiometric_Rlvl6_Llvl2_premixed_730,640,561,488,405_2000,200,300,400,400_Pos2.ome.tif", "/groups/CaiLab/personal/Lex/raw/Ratio_metric_repeat/notebook_pyfiles/max_projected/EEF2_ratiometric_Rlvl6_Llvl3_premixed_730,640,561,488,405_2000,200,300,400,400_Pos2.ome.tif"]

#reference image
ref_img = "/groups/CaiLab/personal/Lex/raw/Ratio_metric_repeat/notebook_pyfiles/max_projected/EEF2_ratiometric_Rlvl6_Llvl2_premixed_730,640,561,488,405_2000,200,300,400,400_Pos2.ome.tif"

#calculate transform
_, _, tform = chromatic_corr_offsets(ref_img, threshold_abs=500,  max_dist=3, ransac_threshold = 0.5, swapaxes=True)

#apply offsets
apply_chromatic_corr(files, tform, cores = 12,  swapaxes=True, write = True)
