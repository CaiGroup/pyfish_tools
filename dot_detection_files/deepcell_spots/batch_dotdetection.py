from deepcell_spots_detection import find_spots_all
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/aberration_corrected/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#This list should contain all hybcycles for a given pos
img_list = files
#Probability threshold to be considered a spot (parameter of deep cell)
probability_threshold = 0.85
#Cutoff for dot size beyond size_cutoff*std (from fitted gaussian)
size_cutoff = 4
#Path for output folder. Code will automatically make the folder.
output_folder = "/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/dots_detected_deepcell"
#If your data set is within channel encoded then set this true.
encoded_within_channel = False
#run parallel processing
parallel = True

if __name__ == "__main__":
    find_spots_all(img_list, probability_threshold, size_cutoff, output_folder, encoded_within_channel, parallel)