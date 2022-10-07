#custom dot detection with deepcell integration
from deepcell_spots_detection import find_spots_all
#file management
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
#feed in input arguments
import sys

#collect inputs
input1 = int(sys.argv[1])
input2 = int(sys.argv[2])

#get job id enviromental variable
JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path to images
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/fiducial_aligned/")

#get all positions for a specific hyb
files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + '/MMStack_Pos{pos}.ome.tif')
files = [str(f) for f in files]
#use inputs from python submission for number of positions
files = files[input1:input2]

#This list should contain all hybcycles for a given pos
img_list = files
#Probability threshold to be considered a spot (parameter of deep cell)
probability_threshold = 0.85
#Cutoff for dot size beyond size_cutoff*std (from fitted gaussian)
size_cutoff = 4
#Path for output folder. Code will automatically make the folder.
output_folder = "/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dots_detected/deepcell"
#If your data set is within channel encoded then set this true.
encoded_within_channel = True
#if performing job by hyb set this to true for file organization
job_by_hyb = True


find_spots_all(img_list, probability_threshold, size_cutoff, output_folder, encoded_within_channel, job_by_hyb)



