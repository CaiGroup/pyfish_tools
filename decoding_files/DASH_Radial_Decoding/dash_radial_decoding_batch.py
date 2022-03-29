from dash_radial_decoding import dash_radial_decoding
import os
import pandas as pd
import numpy as np
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#name of codebooks
codebooks = ["codebook_string_750.csv", "codebook_string_647.csv", "codebook_string_561.csv", "codebook_string_488.csv"]

#collect pos and channel info
pos = int(sys.argv[1])
channel = int(sys.argv[2])

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/031322_11kgenes_experiment/notebook_pyfiles/dots_comb/Channel_{channel}/MMStack_Pos{pos}/Threshold_{JOB_ID}/Dot_Locations/locations_z_0.csv")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/Lex/raw/031322_11kgenes_experiment/barcode_key/{codebooks[channel-1]}"
#number of readout sites
num_barcodes = 3
#search radii
first_radius = 1
second_radius = 1
third_radius = 1.5
#how many allowed drops in calls
diff = 0
#how many times does a pseudocolor sequence must appear
min_seed = 2
#how many times does pseudocolor sequence must appear for highlky expressed genes
high_exp_seed=2
#number of total hybs
hybs = 45
#do you want locations of dots that didn't pass parity
include_undecoded = False
#do you want to decode highly expressed genes first
decode_high_exp_genes = True
#do you want to perform an additional thrid round of decoding
triple_decode = True
#Where do you want to output the files
output_dir = f"/groups/CaiLab/personal/Lex/raw/031322_11kgenes_experiment/notebook_pyfiles/decoded/screen_2_2/Channel_{channel}/Threshold_{JOB_ID}/Pos_{pos}"

if len(locations_path) > 1:
    for locations in locations_path:
        dash_radial_decoding(location_path=locations, codebook_path=codebook_path,
                             num_barcodes=num_barcodes, first_radius=first_radius, 
                             second_radius=second_radius,third_radius=third_radius,
                             diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                             output_dir=output_dir, include_undecoded=include_undecoded, 
                             decode_high_exp_genes=decode_high_exp_genes,
                             triple_decode=triple_decode)
else:
    dash_radial_decoding(location_path=locations_path[0], codebook_path=codebook_path,
                         num_barcodes=num_barcodes, first_radius=first_radius, second_radius=second_radius,
                         third_radius=third_radius, diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                         output_dir=output_dir, include_undecoded=include_undecoded,
                         decode_high_exp_genes=decode_high_exp_genes,
                         triple_decode=triple_decode)

