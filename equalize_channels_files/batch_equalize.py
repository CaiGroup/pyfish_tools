from equalize import equalize_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/dapi_aligned")
position_name = '/MMStack_Pos0.ome.tif'

#get one of the hybs for one position
files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + position_name)
#get hyb 0 with same position
ref, _, _ = find_matching_files(directory, 'HybCycle_0' + position_name) 

img_ref = str(ref[0])
img=str(files[0]) 
across_ch=False
equalize_parallel(img_ref, img, across_ch)
