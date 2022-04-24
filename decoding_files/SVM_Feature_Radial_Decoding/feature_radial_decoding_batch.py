from svm_feature_radial_decoding import feature_radial_decoding
import os
import pandas as pd
import numpy as np
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#name of codebooks
codebooks = ["codebook_string_561.csv"]

#collect pos and channel info
#pos = int(sys.argv[1])
#channel = int(sys.argv[2])

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/notebook_pyfiles/dots_comb/final_thresh7_features/Channel_1/genes_in_cells/MMStack_Pos{JOB_ID}/gene_locations_z_0.csv")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/barcode_key/convert_barcode_key/{codebooks[0]}"
#number of readout sites
num_barcodes = 3
#search radii
first_radius = 1
second_radius = 1.5
third_radius = 2
#how many allowed drops in calls 
diff = 0
#how many times does a pseudocolor sequence must appear
min_seed = 3
#how many times does pseudocolor sequence must appear for highly expressed genes
high_exp_seed=3
#number of total hybs
hybs = 18
#probability cutoff for On dots (0-1). Lower the value the less stringent.
probability_cutoff = 0.25
#desired FDR (0-1). Could set to None if you would like to filter yourself.
desired_fdr = 0.05
#do you have parity round
parity_round = False
#do you want locations of dots that didn't pass parity
include_undecoded = False
#do you want to decode highly expressed genes first
decode_high_exp_genes = True
#do you want to perform an additional third round of decoding
triple_decode = True
#Where do you want to output the files
output_dir = f"/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/notebook_pyfiles/decoded/final_thresh7_11p52_33_heg_svm_test/Channel_1/Pos_{JOB_ID}"

if len(locations_path) > 1:
    for locations in locations_path:
        feature_radial_decoding(location_path=locations, codebook_path=codebook_path,
                             num_barcodes=num_barcodes, first_radius=first_radius, 
                             second_radius=second_radius,third_radius=third_radius,
                             diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                             probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                             output_dir=output_dir, include_undecoded=include_undecoded, 
                             decode_high_exp_genes=decode_high_exp_genes,
                             triple_decode=triple_decode, parity_round=parity_round)
else:
    feature_radial_decoding(location_path=locations_path[0], codebook_path=codebook_path,
                         num_barcodes=num_barcodes, first_radius=first_radius, second_radius=second_radius,
                         third_radius=third_radius, diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                         probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                         output_dir=output_dir, include_undecoded=include_undecoded,
                         decode_high_exp_genes=decode_high_exp_genes,
                         triple_decode=triple_decode, parity_round=parity_round)

