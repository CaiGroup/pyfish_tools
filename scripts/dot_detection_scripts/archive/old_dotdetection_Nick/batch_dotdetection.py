from dot_detection import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/sliced_img")
position_name = '/MMStack_Pos0.ome.tif'

files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + position_name)
files = str(files[0])

num_channels = 3

dot_detection_parallel(files, num_channels)

