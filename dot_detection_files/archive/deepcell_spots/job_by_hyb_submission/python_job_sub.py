#file management
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import numpy as np

#general path to images
directory = Path("/groups/CaiLab/personal/Lex/raw/031322_11kgenes_experiment/notebook_pyfiles/pre_processed_images/")

#get all positions for a specific hyb
files, _, _ = find_matching_files(directory, 'HybCycle_0' + '/MMStack_Pos{pos}.ome.tif')
files = [str(f) for f in files]

#how many positions
num_pos = len(files)
#first check if the number of position is greater than or equal to 12
if num_pos > 12:
    #break up into intervals of 12 positions
    chunks = np.linspace(0,num_pos,round(num_pos/12)).astype(int)
else:
    chunks = [0,12]
    
#submit multipls jobs for different positions
for i in range(len(chunks)-1):
    #write batch files
    job_file = f"dotdetection_chunk{chunks[i]}_{chunks[i+1]}.batch"
    with open(job_file, "w+") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --ntasks=1\n") #num tasks keep at 1
        f.write("#SBATCH --mem=100G\n") # RAM amount
        f.write("#SBATCH --cpus-per-task=30\n") #number of cpus
        f.write("#SBATCH --time=03:00:00\n") #how much time
        f.write("#SBATCH --array=0-45\n") #hyb range
        f.write("pwd; hostname; date\n") #output path, hostname, date
        f.write("echo This is task $SLURM_ARRAY_TASK_ID\n") #output task id
        f.write("source ~/miniconda3/bin/activate\n") #activate source
        f.write("conda activate python3.7\n") #activate conda env
        f.write(f"python batch_dotdetection.py {chunks[i]} {chunks[i+1]}\n") #run batch.py and feed in arguments
        f.write("date\n")
        f.close()

    os.system("sbatch %s" %job_file)