import numpy as np
import os

###Here is a general template for multi parameter job submission for decoding

#number of pos
pos_list = np.arange(0,5,1)
#number of channels
channels = [1,2]

#submit multipls jobs for different positions
for pos in pos_list:
    for channel in channels:
    #write batch files
        job_file = f"decoding_pos{pos}_ch{channel}.batch"
        with open(job_file, "w+") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --ntasks=1\n") #num tasks keep at 1
            f.write("#SBATCH --mem=25G\n") # RAM amount
            f.write("#SBATCH --cpus-per-task=12\n") #number of cpus
            f.write("#SBATCH --time=00:30:00\n") #how much time
            f.write("#SBATCH --array=0-11\n") #position list or threshold (depends on intentions)
            f.write("pwd; hostname; date\n") #output path, hostname, date
            f.write("echo This is task $SLURM_ARRAY_TASK_ID\n") #output task id
            f.write("source ~/miniconda3/bin/activate\n") #activate source
            f.write("conda activate python3.7\n") #activate conda env
            f.write(f"python nonbarcoded_smfish_decode_batch.py {pos} {channel}\n") #run batch.py and feed in arguments
            f.write("date\n")
            f.close()

        os.system("sbatch %s" %job_file)
