from svm_feature_radial_decoding_v3 import decode
import os
import glob
import sys
JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)
#____________________________________________________________________________________________________________________________
channel                      = int(sys.argv[1]) #collect channel info from params file
codebooks                    = ["codebook_string_647.csv", "codebook_string_532.csv", "codebook_string_488.csv"] #name of codebooks
exp_dir                      = "Linus_10k_cleared_080918_NIH3T3" #name of experimental directory
user                         = "Lex" #name of user
num_rounds                   = 4     #number of rounds
first_radius                 = 1.0   #first search radii
second_radius                = 1.4   #second search radii
third_radius                 = 2     #third search radii if applicable
min_seed                     = 3     #how many times does a pseudocolor sequence must appear
high_exp_seed                = 3     #how many times does pseudocolor sequence must appear for highly expressed genes
total_hybs                   = 80    #number of total hybs
probability_cutoff           = 0.20  #probability cutoff for On dots (0-1). Lower the value the less stringent.
desired_fdr                  = 0.10  #desired FDR (0-1). Could set to None if you would like to filter yourself.
use_svm                      = True  #set to false if model is not helping and to save time.
parity_round                 = True  #do you have parity round
include_undefined            = False #do you want locations of dots that didn't pass parity
decode_high_exp_genes_first  = True  #do you want to decode highly expressed genes first
triple_decode                = True  #do you want to perform an additional third round of decoding
score_brightness             = True  #do you want to use brightness for scoring 
blank_ch                     = False #will you have a blank channel

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/dots_detected/fiducials_removed/Channel_{channel}/Pos{JOB_ID}/*")
#____________________________________________________________________________________________________________________________
#No need to really adjust unless you want alternative input or output paths

#general codebook path
codebook_path = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/barcode_key/{codebooks[channel-1]}"

if parity_round:
    for diff in range(2):
        #Where do you want to output the files
        output_dir = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/decoded_fid_rem/final_{first_radius}{second_radius}{third_radius}_seed{min_seed}{high_exp_seed}_heg_svm_p{probability_cutoff*100}_diff{diff}_fdr{desired_fdr*100}/Channel_{channel}/Pos_{JOB_ID}"
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
                                     score_brightness = score_brightness, blank_ch=blank_ch)
                decoder.feature_radial_decoding(use_svm = use_svm )
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
                                    score_brightness = score_brightness, blank_ch=blank_ch)
            decoder.feature_radial_decoding(use_svm = use_svm )
else:
    #Where do you want to output the files
    output_dir = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/decoded/final_{first_radius}{second_radius}{third_radius}_seed{min_seed}{high_exp_seed}_heg_svm_p{probability_cutoff*100}_diff{diff}_fdr{desired_fdr*100}/Channel_{channel}/Pos_{JOB_ID}"
    if len(locations_path) > 1:
        for locations in locations_path:
            decoder = decode(location_path=locations, codebook_path=codebook_path,
                                 num_rounds=num_rounds, first_radius=first_radius, 
                                 second_radius=second_radius,third_radius=third_radius,
                                 diff=0, min_seed=min_seed, high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                                 probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                                 output_dir=output_dir, include_undefined=include_undefined, 
                                 decode_high_exp_genes_first=decode_high_exp_genes_first,
                                 triple_decode=triple_decode, parity_round=parity_round, 
                                 score_brightness = score_brightness, blank_ch=blank_ch)
            decoder.feature_radial_decoding(use_svm = use_svm )
    else:
        decoder = decode(location_path=locations_path[0], codebook_path=codebook_path,
                                num_rounds=num_rounds, first_radius=first_radius,
                                second_radius=second_radius,
                                third_radius=third_radius, diff=0, min_seed=min_seed,
                                high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                                probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                                output_dir=output_dir, include_undefined=include_undefined,
                                decode_high_exp_genes_first=decode_high_exp_genes_first,
                                triple_decode=triple_decode, parity_round=parity_round, 
                                score_brightness = score_brightness, blank_ch=blank_ch)
        decoder.feature_radial_decoding(use_svm = use_svm )