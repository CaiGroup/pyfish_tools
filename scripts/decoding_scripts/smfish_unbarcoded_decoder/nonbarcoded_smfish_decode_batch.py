from unbarcoded_smfish_assign import assign_genes_to_spots_parallel
import os
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#collect pos and channel info
pos = int(sys.argv[1])
channel = int(sys.argv[2])

#get codebook path
codebook = "/groups/CaiLab/personal/Lex/raw/042022_40genes_smfish/barcode_key/smfish_key.csv"

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/042022_40genes_smfish/notebook_pyfiles/dots_comb/Channel_{channel}/MMStack_Pos{pos}/Threshold_{JOB_ID}/genes_in_cells/Dot_Locations/*")
                           
#desired output dir
output_dir = f"/groups/CaiLab/personal/Lex/raw/042022_40genes_smfish/notebook_pyfiles/decoded/screen/Channel_{channel}/Threshold_{JOB_ID}/Pos_{pos}"

assign_genes_to_spots_parallel(locations_path, codebook, output_dir)