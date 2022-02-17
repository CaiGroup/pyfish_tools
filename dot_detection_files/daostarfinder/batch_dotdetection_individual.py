from daostarfinder_dotdetection import dot_detection_parallel
from pathlib import Path
import glob
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#path to processed images
directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#pos number from find threshold, just provide the location of MMStack_Posx folders
thresh_pos = glob.glob("/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/threshold_counts/Channel_1/HybCycle_1/*")
pos_list = []
for path in thresh_pos:
    pos = int(Path(path).name.split("_")[1].replace("Pos",""))
    pos_list.append(pos)

HybCycle = JOB_ID
size_cutoff = 3 # sigma cutoff for size distribution
channel = 4 #which channel to analyze
choose_thresh_set = 3 #select best thresh set
hyb_number=12 #total number of hybs
optimize=False #are you testing thresholds
output=True #do you want to write out results

dot_detection_parallel(files, HybCycle,size_cutoff,channel,pos_list,
                       choose_thresh_set,hyb_number, optimize, output)
