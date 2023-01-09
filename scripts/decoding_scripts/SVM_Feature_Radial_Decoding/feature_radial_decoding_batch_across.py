from svm_feature_radial_decoding import decode
import os
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#name of experimental directory
exp_dir = "102222_10K_NIH3T3"
#name of user
user = "Lex"
#number of rounds
num_rounds = 4
#search radii
first_radius = 1
second_radius = 1.5
third_radius = 1.5
#how many allowed drops in calls 
diff = 1
#how many times does a pseudocolor sequence must appear
min_seed = 3
#how many times does pseudocolor sequence must appear for highly expressed genes
high_exp_seed = 3
#number of total hybs
total_hybs = 20
#probability cutoff for On dots (0-1). Lower the value the less stringent. Setting probability_cutoff=0 and desired_fdr=None, will output normal unfiltered data.
probability_cutoff = 0.25
#desired FDR (0-1). Could set to None if you would like to filter yourself.
desired_fdr = 0.15
#do you have parity round
parity_round = True
#do you want locations of dots that didn't pass parity
include_undefined = False
#do you want to decode highly expressed genes first
decode_high_exp_genes_first = True
#do you want to perform an additional third round of decoding
triple_decode = True
#do you want to use brightness for scoring 
score_brightness = True
#____________________________________________________________________________________________________________________________

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/seqFISH_datapipeline/output/dots_detected/Channel_All/Pos{JOB_ID}/*")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/barcode_key/codebook_string_across.csv"
#Where do you want to output the files
output_dir = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/seqFISH_datapipeline/output/decoded/final_{first_radius}{second_radius}{third_radius}_seed{min_seed}{high_exp_seed}_heg_svm_p{probability_cutoff*100}_diff{diff}_fdr{desired_fdr*100}/Channel_All/Pos_{JOB_ID}"

if len(locations_path) > 1:
    for locations in locations_path:
        decoder = decode(location_path=locations, codebook_path=codebook_path,
                             num_rounds=num_rounds, first_radius=first_radius, 
                             second_radius=second_radius,third_radius=third_radius,
                             diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                             probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                             output_dir=output_dir, include_undefined=include_undefined, 
                             decode_high_exp_genes_first=decode_high_exp_genes_first,
                             triple_decode=triple_decode, parity_round=parity_round, 
                             score_brightness = score_brightness)
        decoder.feature_radial_decoding()
else:
    decoder = decode(location_path=locations_path[0], codebook_path=codebook_path,
                            num_rounds=num_rounds, first_radius=first_radius,
                            second_radius=second_radius,
                            third_radius=third_radius, diff=diff, min_seed=min_seed,
                            high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                            probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                            output_dir=output_dir, include_undefined=include_undefined,
                            decode_high_exp_genes_first=decode_high_exp_genes_first,
                            triple_decode=triple_decode, parity_round=parity_round, 
                            score_brightness = score_brightness)
    decoder.feature_radial_decoding()

