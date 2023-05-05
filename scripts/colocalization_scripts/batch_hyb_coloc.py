from hyb_coloc import coloc_parallel
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to processed images
img_dir = ""

channel = int(JOB_ID) # which channel to analyze
z=2 # which z layer to focus on
threshold = 0.05 # intensity threshold 
radii_list = [0.75,1,2] # various radii to test
num_pos = 25 # number of pos
hyb_list = [0,8] # two hybs that should colocalize
num_channels = 4 # bool to flip z and channel axis


coloc_parallel(img_dir = img_dir, channel = channel, z = z, threshold = threshold, 
                   radii_list = radii_list, num_pos = num_pos,
                   hyb_list = hyb_list, num_channels = num_channels)