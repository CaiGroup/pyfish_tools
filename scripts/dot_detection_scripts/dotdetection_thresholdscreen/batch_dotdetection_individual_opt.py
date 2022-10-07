from daostarfinder_dotdetection_screen import dot_detection_parallel
import sys
from pathlib import Path
import glob
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#collect pos and channel info
pos = int(sys.argv[1])
channel = int(sys.argv[2])

#path for pos
directory = Path("/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{pos}.ome.tif'
file = str(directory / f"HybCycle_{JOB_ID}"/ position_name) 

#pos number from find threshold, just provide the location of MMStack_Posx folders
#channel doesn't matter here
thresh_pos = glob.glob(f"/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/notebook_pyfiles/threshold_counts/Channel_{channel}/HybCycle_{JOB_ID}/*")
pos_list = []
for path in thresh_pos:
    pos = int(Path(path).name.split("_")[1].replace("Pos",""))
    pos_list.append(pos)
    
#arguments
HybCycle = JOB_ID
size_cutoff = 4 # sigma cutoff for size distribution
channel = channel #which channel to analyze (1-4)
choose_thresh_set = 0 #ignore for optimization
optimize = True #are you testing thresholds
output = True #do you want to write out results
                 
dot_detection_parallel(file, HybCycle, size_cutoff,channel,pos_list,
                       choose_thresh_set,
                       optimize, output)
