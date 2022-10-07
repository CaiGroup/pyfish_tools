import numpy as np
import os

###Here is a general template for multi parameter job submission for decoding

#number of pos
pos_list = np.arange(0,5,1)
#number of channels
channel_list = np.arange(1,5,1)
#optional info to add to batch name
seed=2

#submit multipls jobs for different positions
for pos in pos_list:
    for channel in channel_list:
        #write batch files
        job_file = f"decoding_pos{pos}_ch{channel}_seed{seed}.batch"
        with open(job_file, "w+") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --ntasks=1\n") #num tasks keep at 1
            f.write("#SBATCH --mem=50G\n") # RAM amount
            f.write("#SBATCH --cpus-per-task=3\n") #number of cpus
            f.write("#SBATCH --time=16:00:00\n") #how much time
            f.write("#SBATCH --array=0-11\n") #position list or threshold (depends on intentions)
            f.write("pwd; hostname; date\n") #output path, hostname, date
            f.write("echo This is task $SLURM_ARRAY_TASK_ID\n") #output task id
            f.write("source ~/miniconda3/bin/activate\n") #activate source
            f.write("conda activate python3.7\n") #activate conda env
            f.write(f"python feature_radial_decoding_batch.py {pos} {channel}\n") #run batch.py and feed in arguments
            f.write("date\n")
            f.close()

        os.system("sbatch %s" %job_file)
