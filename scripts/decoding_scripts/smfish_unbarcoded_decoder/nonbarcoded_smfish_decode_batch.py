from unbarcoded_smfish_assign import assign_genes_to_spots_parallel
import os
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#collect pos and channel info
pos = int(sys.argv[1])
channel = int(sys.argv[2])

#get codebook path
codebook = ""

#path to dots
locations_path = glob.glob(f"")
                           
#desired output dir
output_dir = f""

assign_genes_to_spots_parallel(locations_path, codebook, output_dir)