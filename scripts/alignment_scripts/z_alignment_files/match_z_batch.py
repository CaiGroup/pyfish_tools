#custom function
from match_z_across_hybs import z_matching
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to images
image_dir = "/groups/CaiLab/personal/Lex/raw/230708_3k_1nM_split_IP_tmg/pyfish_tools/output/dapi_aligned"

#run script
z_matching(image_dir, pos_number = JOB_ID)
