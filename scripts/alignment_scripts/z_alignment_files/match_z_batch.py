#custom function
from match_z_across_hybs import z_matching
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to images
image_dir = ""

#run script
z_matching(image_dir, pos_number = JOB_ID)
