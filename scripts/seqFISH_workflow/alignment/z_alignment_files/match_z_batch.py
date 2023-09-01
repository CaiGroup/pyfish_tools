#custom function
from match_z_across_hybs import z_matching
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to images
image_dir = "/groups/CaiLab/personal/Lex/raw/230812_3k_IP_4bs/pyfish_tools/output/dapi_aligned"
#reference dir
ref_dir = "/groups/CaiLab/personal/Lex/raw/230812_3k_IP_4bs/chromatic_aberration"
#number of channels
num_channels = 4

#run script
z_matching(image_dir, ref_dir, num_channels = num_channels, pos_number = JOB_ID)
