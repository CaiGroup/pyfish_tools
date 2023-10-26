from hyb_coloc import coloc_parallel
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to processed images
img_dir      = "/path/to/data/pyfish_tools/output/pre_processed_images"
channel      = int(JOB_ID) # which channel to analyze
z            = 1 # which z layer to focus on
threshold    = 0.2 # intensity threshold 
radii_list   = [0.75,1,2] # various radii to test
num_pos      = 50 # number of pos
hyb_list     = [0,20] # two hybs that should colocalize
num_channels = 4 # used for having correct axes

coloc_parallel(img_dir = img_dir, channel = channel, z = z, threshold = threshold, 
                   radii_list = radii_list, num_pos = num_pos,
                   hyb_list = hyb_list, num_channels = num_channels)