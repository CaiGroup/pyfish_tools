"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 03/21/22
"""

#general analysis packages
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
#parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
#path management
from pathlib import Path
import sys


def filter_dots_fast(dots_idx, min_seed=4):
    """Keep unique sets that appear >= min_seed times.
    Parameters
    ----------
    dots_idx = list of dot sets as tuples
    min_seed = number of times a dot sequence should appear
    
    Returns
    --------
    filtered_set = dots that appeared >= min_seed times
    """
    
    #generate frozen set
    flat_list = [frozenset(element) for element in dots_idx]
    #count the number of times a frozen set appeared
    obj = Counter(flat_list)
    #convert to df
    df = pd.DataFrame.from_dict(obj, orient='index').reset_index()
    #keep sets that appear >= the min seed
    filtered_set = df[df[0] >= min_seed]["index"].to_list()
    
    #return unique sets
    return filtered_set

def score_codewords(code_list,codeword_hash, distance_hash):
    """
    This function will pick best overall codeword
    Parameters
    ----------
    code_list: list of codword sets we are trying to compare
    codeword_hash: a hash table to return overall codeword score
    distance_hash: a hash table to return overall distance for codeword
    """
    #keep score and distance for each dot sequence
    track_score = []
    track_distance = []
    for codeword in code_list:
        track_score.append(codeword_hash[tuple(sorted(codeword))])
        track_distance.append(distance_hash[tuple(sorted(codeword))])
    #check if there are equal scores
    #if that is the case use pure distance (taking the shortest)
    if len(np.where(track_score == np.amax(track_score))[0]) >= 2:
        best_code = code_list[np.argmin(track_distance)]
    else:
        best_code = code_list[np.argmax(track_score)]
    
    return best_code

def recover_codes(code_list, best_codeword):
    """
    This function will try to recover codewords if the dots selected 
    from score_codewords() function is not repeated in the other choices
    Parameters
    ----------
    code_list: list of codeword sets we analyzed from score_codewords()
    best_codeword: the best codeword set chosen from score_codewords()
    """
    #store winner history
    winner_history = set()
    for unique_dots in best_codeword:
        winner_history.add(unique_dots)
    #recover codewords if the winner dot sequence does not appear in other choices
    recovered_codes = []
    for i in range(len(code_list)):
        if code_list[i] & winner_history:
            continue
        else:
            recover_code = code_list[i]
            recovered_codes.append(recover_code)
            
    return recovered_codes

def pick_best_codeword(filtered_set, codeword_scores,total_distances):
    """This function will try to pick between multiple codewords based on codeword
    score generated from radial_decoding function and overall codeword distance.
    If there is only one choice, then it will just use that.
    
    Parameters
    ----------
    filtered_set = ouput from filter_dots_fast
    codeword_scores = output from radial decoding
    total_distances = output from radial decoding
    
    Returns
    ----------
    complete_set = final set of dots that are unique and high scoring
    """
    
    #group all sets with any matching dot indicies
    matching_element_list = []
    #make a copy of list for searching
    search_set = filtered_set.copy()
    #keep history of elements
    history_element = set()
    #lets group
    for _set in filtered_set:
        temp=[]
        if _set & history_element:
            continue
        for _search in search_set:
            if _set & _search:
                temp.append(_search)
        for item in np.unique(list(_set)):
            history_element.add(item)
        matching_element_list.append(temp)

    #isolate only lists > 1
    filter_list = []
    for sublist in matching_element_list:
        if len(sublist) > 1:
            filter_list.append(sublist)
            
    #generate hash table for codeword score
    codeword_hash = {}
    for i in range(len(codeword_scores)):
        #if the key exists, add the codeword score to existing key
        if tuple(sorted(codeword_scores[i][0])) in codeword_hash:
            new_value = codeword_hash[tuple(sorted(codeword_scores[i][0]))] + codeword_scores[i][1]
            codeword_hash.update({tuple(sorted(codeword_scores[i][0])):new_value})
        else:
            codeword_hash.update({tuple(sorted(codeword_scores[i][0])):codeword_scores[i][1]})
            
    #generate hash table for total distance
    distance_hash = {}
    for i in range(len(total_distances)):
        #if the key exists, add the total distance to existing key
        if tuple(sorted(total_distances[i][0])) in distance_hash:
            new_value = distance_hash[tuple(sorted(total_distances[i][0]))] + total_distances[i][1]
            distance_hash.update({tuple(sorted(total_distances[i][0])):new_value})
        else:
            distance_hash.update({tuple(sorted(total_distances[i][0])):total_distances[i][1]})
            
    #pick the best final codewords
    final_codewords = []
    for codes in filter_list:
        #get best code word from list of sets
        best_code = score_codewords(codes,codeword_hash, distance_hash)
        #store winner
        final_codewords.append(list(best_code))
        #check to see if we can recover any codewords
        recovered_codes = recover_codes(codes, best_code)
        #if there is one returned, we will add to final
        if len(recovered_codes) == 1:
            final_codewords.append(list(recovered_codes[0]))
        #if nothing is returned then we will continue
        elif len(recovered_codes) == 0:
            continue
        #if there is multiple choices we will perform a while loop until the choices become 0
        else:
            while len(recovered_codes) != 0:
                #find the best codeword in recovered list
                best_2 = score_codewords(recovered_codes,codeword_hash, distance_hash)
                final_codewords.append(list(best_2))
                #check to see if we can recover any codewords again
                recovered_2 = recover_codes(recovered_codes, best_2)
                #if there is one choice returned we can add
                #otherwise we will repeat loop or end loop
                if len(recovered_2) == 1:
                    final_codewords.append(list(recovered_2[0]))
                #reassign recovered_codes
                recovered_codes = recovered_2      
    
    #get the truly unique ones from matching_element_list
    filter_list_unique = []
    for sublist in matching_element_list:
        if len(sublist) == 1:
            filter_list_unique.append(list(sublist[0]))
    
    #combine the two unique sets
    complete_set = filter_list_unique+final_codewords 
    
    #delete some variables
    del matching_element_list
    del filter_list
    del search_set
    del history_element
    del codeword_hash
    del distance_hash
    del final_codewords
    del filter_list_unique
    
    return complete_set
    
def radial_decoding(locations,num_barcodes = 4, 
                    radius=np.sqrt(2),diff=0,
                    seed=0, hybs = 12, decode_across = False):
    """
    
    This function will decode dots utilizing kNN algorithm from sklearn with a euclidean distance 
    metric. Essentially, a defined search radius will be used to identify nearby dots. Dot sequence will be
    determined using a score based metric incorporating distance, intensity, and size. Dot sequences that appear n
    number of time defined by min seed will be kept. A table will be generated for each unnamed
    gene, at which a codebook will act as a hash table for identification. This decoder should operate
    similarly to MATLAB decoding. However, this decoder can be used for seqFISH DASH datasets, and will try to
    assign ambiguous dots.
    
    Parameters
    ----------
    locations = locations.csv file
    num_barcodes = number of total barcodes
    radius = search radius using euclidean metric
    diff = allowed barcode drops
    seed = which barcode set to use as reference
    hybs = total number of hybs
    decode_across = bool for decoding across
    
    Returns
    --------
    dot indicies that were determined to be the best sequence
    
    """
    #using sklearn nearest neighbor algorithm to find nearest dots
    #initialize algorithm
    neigh = NearestNeighbors(n_neighbors=num_barcodes, radius=radius, metric="euclidean", n_jobs=1)
    
    barcoding_round = []
    #separate locations by barcoding round
    hyb_rounds = np.arange(0,hybs,1)
    temp = []
    for h in hyb_rounds:
        if h == hyb_rounds[len(hyb_rounds)-1]:
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)
            comp_round = pd.concat(temp)
            barcoding_round.append(comp_round) 
        elif (h % (hybs/num_barcodes) != 0) or (h == 0):
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)
        else:
            comp_round = pd.concat(temp)
            barcoding_round.append(comp_round)
            temp = []
            barcode = locations[locations["hyb"] == h]
            temp.append(barcode)
            
    #remove temp list
    del temp
    
    #store index of barcoding round that is being compared against in list
    initial_barcoding_index = barcoding_round[seed][["x","y"]].index.tolist()
    
    #rename seed
    initial_seed = barcoding_round[seed][["x","y"]]
    
    #delete barcoding round that is being compared against in list
    del barcoding_round[seed]
    
    #find neighbors at a given search radius between barcoding rounds
    neighbor_list = []
    distance_list = []
    for i in range(len(barcoding_round)):
        #initialize neighbor
        neigh.fit(barcoding_round[i][["x","y"]])
        #find neighbors for barcoding round 0
        distances,neighbors = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
        distance_list.append(distances)
        neighbor_list.append(neighbors)
    
    #organize dots so that each barcode is pooled together
    neighbor_list2 = []
    distance_list2 = []
    for i in range(len(neighbor_list[0])):
        temp = []
        temp_dist =[]
        #loop through barcoding rounds for same position
        for j in range(len(neighbor_list)):
            temp.append(neighbor_list[j][i].tolist())
            temp_dist.append(distance_list[j][i].tolist())
        #convert to original index on df
        orig_idx = []
        for _ in range(len(temp)):
            barcode_pool = []
            for idx in temp[_]:
                try:
                    barcode_pool.append(barcoding_round[_].iloc[idx].name)
                except IndexError:
                    barcode_pool.append([])
            orig_idx.append(barcode_pool)
        #add index of dot being analyzed
        orig_idx.insert(0,[initial_barcoding_index[i]])
        #remove empty lists which correspond to missed dots in barcoding round
        orig_idx = [sublist for sublist in orig_idx if sublist != []]
        total_distance = [sublist for sublist in temp_dist if sublist != []]
        #remove arrays less than allowed diff
        #there will be a -1 for distance array because we wouldn't have distance value for self
        if len(orig_idx) >= (num_barcodes-diff):
            neighbor_list2.append(orig_idx)
        if len(total_distance) >= (num_barcodes-diff-1):
            distance_list2.append(total_distance)
        else:
            continue  
    #remove old lists
    del neighbor_list
    del distance_list
    
    #Get best dot sequence based on dot traits
    #weighted scores will be calculated based on distance, flux, and size
    dot_idx = []
    ambiguity_scores = []
    codeword_score_list = []
    total_distance_list = []
    for i in range(len(neighbor_list2)):
        temp = []
        ambiguity = 0
        codeword_score = 0
        total_distance_values = 0
        for j in range(len(neighbor_list2[i])):
            if len(neighbor_list2[i][j]) == 1:
                temp.append(neighbor_list2[i][j][0])
                ambiguity += 1
                codeword_score += 1
                if j == 0:
                    continue
                else:
                    total_distance_values += distance_list2[i][j-1][0]
            else:
                if decode_across == False:
                    #generate scoring system
                    dist_values = (np.linspace(0,2,len(neighbor_list2[i][j])+1)/2)[::-1]
                    int_values = (np.linspace(0,1,len(neighbor_list2[i][j])+1)/2)[::-1]
                    size_values = (np.linspace(0,0.5,len(neighbor_list2[i][j])+1)/2)[::-1]
                    #get dot traits
                    #note that pandas iloc preserve order matching slice order
                    dot_traits = locations.iloc[neighbor_list2[i][j]]
                    #generate score table
                    trait_score = np.zeros(len(dot_traits))
                    #rank dots and see if we are using flux or average intensity
                    try:
                        int_score = np.argsort(dot_traits["flux"]).values[::-1]
                        size_score = np.argsort(dot_traits["size"]).values[::-1]
                    except:
                        int_score = np.argsort(dot_traits["average intensity"]).values[::-1]
                        size_score = np.argsort(dot_traits["size"]).values[::-1]
                    #calculate best score
                    #note that distance is already sorted
                    for _ in range(len(trait_score)):
                        trait_score[int_score[_]] += int_values[_]
                        trait_score[_] += dist_values[_]
                        trait_score[size_score[_]] += size_values[_]
                    #get highest scoring index, then get all scores
                    best_dot_idx = np.argmax(trait_score)
                    temp.append(neighbor_list2[i][j][best_dot_idx])
                    ambiguity += len(neighbor_list2[i][j])
                    codeword_score += trait_score[best_dot_idx]
                    total_distance_values += distance_list2[i][j-1][best_dot_idx]
                else:
                    #int and size will have no contribution to choice only distance
                    dist_values = (np.linspace(0,2,len(neighbor_list2[i][j])+1)/2)[::-1]
                    #get dot traits
                    #note that pandas iloc preserve order matching slice order
                    dot_traits = locations.iloc[neighbor_list2[i][j]]
                    #generate score table
                    trait_score = np.zeros(len(dot_traits))
                    #calculate best score
                    #note that distance is already sorted
                    for _ in range(len(trait_score)):
                        trait_score[_] += dist_values[_]
                    #get highest scoring index, then get all scores
                    best_dot_idx = np.argmax(trait_score)
                    temp.append(neighbor_list2[i][j][best_dot_idx])
                    ambiguity += len(neighbor_list2[i][j])
                    codeword_score += trait_score[best_dot_idx]
                    total_distance_values += distance_list2[i][j-1][best_dot_idx]
        #calculate final scores
        ambiguity_score_final = ambiguity-len(neighbor_list2[i])
        codeword_score -= ambiguity_score_final 
        #append all of the data to final list
        dot_idx.append(temp)
        ambiguity_scores.append([tuple(temp),ambiguity_score_final])
        codeword_score_list.append([tuple(temp),codeword_score])
        total_distance_list.append([tuple(temp),total_distance_values])
        
    #no longer need this  
    del neighbor_list2
    del distance_list2
    del temp
    
    #return indicies of nearby dots from seed, ambiguity scores, codeword scores and total distance per codeword
    return dot_idx,ambiguity_scores,codeword_score_list,total_distance_list

def radial_decoding_parallel(locations,codebook,num_barcodes = 4, radius=1,diff=0,
                             min_seed=4, hybs = 12, include_undecoded = False, decode_across = False):
    """This function will perform radial decoding on all barcodes as reference. Dot sequences
    that appear n number of times defined by min seed will be kept.
    Parameters 
    ----------
    locations = location.csv file
    codebook = codebook.csv
    num_barcodes = number of total barcodes
    radius = search radius using euclidean metric
    diff = allowed barcode drops
    min_seed = number of barcode seeds
    hybs = total number of hybs
    include_undecoded = bool to output the undecoded dots
    decode_across = bool for decoding across
    
    Returns
    --------
    gene_locations.csv
    """
    
    #make sure diff is not greater than 1
    assert diff < 2, "Diff cannot be > 1"
    
    #parallel processing for nearest neighbor computation for each barcode
    with ProcessPoolExecutor(max_workers=num_barcodes) as exe:
        futures = []
        for i in range(num_barcodes):
            seed = i
            fut = exe.submit(radial_decoding, locations,
                             num_barcodes, radius,diff,
                             seed, hybs, decode_across)
            futures.append(fut)
            
    #collect result from futures objects
    result_list = [fut.result() for fut in futures]
    #split dot index list, ambiguity list, codeword score list and total distance while flattening them
    dot_index_list = [element for sublist in result_list for element in sublist[0]]
    ambiguity_list = [element for sublist in result_list for element in sublist[1]]
    codeword_score_list = [element for sublist in result_list for element in sublist[2]]
    total_distance_list = [element for sublist in result_list for element in sublist[3]]
    
    #keep dots that match min seed
    dot_idx_min = filter_dots_fast(dot_index_list, min_seed=min_seed)
        
    #pick best codewords where there could be multiple options
    dot_idx_filtered =  pick_best_codeword(dot_idx_min, codeword_score_list, total_distance_list)
    
    #generate hash table for amiguity score assignment
    amb_dict = {}
    for i in range(len(ambiguity_list)):
        #if the key exists, add the ambiguity score to existing key
        if tuple(sorted(ambiguity_list[i][0])) in amb_dict:
            new_value = amb_dict[tuple(sorted(ambiguity_list[i][0]))] + ambiguity_list[i][1]
            amb_dict.update({tuple(sorted(ambiguity_list[i][0])):new_value})
        else:
            amb_dict.update({tuple(sorted(ambiguity_list[i][0])):ambiguity_list[i][1]})
        
    #make list of ambiguity scores for each dot sequence index
    ambiguity_scores_final =[]
    for i in range(len(dot_idx_filtered)):
        score = amb_dict[tuple(sorted(dot_idx_filtered[i]))]
        ambiguity_scores_final.append(score)
    
    #isolate dot info for barcode sets
    dot_info = []
    for idx in dot_idx_filtered:
        dot_info.append(locations.iloc[idx])
        
    #generate code table for decoding and store info about dots
    info_list = []
    code_table = np.zeros(shape=(len(dot_info), hybs)).astype(int)
    for i in range(len(dot_info)):
        code = dot_info[i][["hyb","ch"]].values
        try:
            info = dot_info[i][["x","y","z","flux","max intensity","size"]].mean().values
        except:
            info = dot_info[i][["x","y","z","average intensity","peak intensity","size"]].mean().values
        info_list.append(info)
        for j in range(len(code)):
            code_table[i][int(code[j][0])] = int(code[j][1])
    
    if diff == 0:
        #convert codebook to hash table
        hash_table = {}
        for i in range(len(codebook)):
            codeword = {tuple(codebook.iloc[i].to_list()):codebook.iloc[i].name}
            hash_table.update(codeword)

        #using hash table for decoding
        decoded_genes = []
        for i in range(len(code_table)):
            try:
                gene_name = hash_table[tuple(code_table[i])]
                decoded_genes.append(gene_name)
            except KeyError:
                sys.exit("Check codebook for completeness")

        #add gene names
        genes_locations = pd.DataFrame(info_list)
        genes_locations.columns = ["x","y","z","brightness","peak intensity","size"]
        genes_locations["genes"] = decoded_genes
        #add ambiguity score
        genes_locations["ambiguity score"] = ambiguity_scores_final
        genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity", "size", "ambiguity score"]]  

    elif diff == 1:
        #make other possible codebooks
        potential_codebooks = []
        for _ in range(len(codebook)):
            new_codewords = []
            new_codewords.append(codebook.iloc[_].values)
            for i in range(num_barcodes):
                codeword = np.copy(codebook.iloc[_].values)
                locations_adj = np.argwhere(codeword > 0)
                codeword[locations_adj[i]] = 0
                new_codewords.append(codeword)
            potential_codebooks.append(new_codewords)
        
        #add codebooks to hash table
        hash_table = {}
        for i in range(len(potential_codebooks)):
            for j in range(len(potential_codebooks[i])):
                codeword = {tuple(potential_codebooks[i][j].tolist()):codebook.iloc[i].name}
                hash_table.update(codeword)
                
        #using hash table for decoding
        decoded_genes = []
        for i in range(len(code_table)):
            try:
                gene_name = hash_table[tuple(code_table[i])]
                decoded_genes.append(gene_name)
            #these sets of key error codes can arise if error correction created a codeword beyond the codebook
            #these are essentially uncorrectable
            except KeyError:
                decoded_genes.append("Undefined")
                
        #make final df
        genes_locations = pd.DataFrame(info_list)
        genes_locations.columns = ["x","y","z","brightness","peak intensity","size"]
        genes_locations["genes"] = decoded_genes
        genes_locations["ambiguity score"] = ambiguity_scores_final
        if include_undecoded ==  False:
            genes_locations = genes_locations[genes_locations["genes"] != "Undefined"]
        genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity", "size", "ambiguity score"]]  
        
    #return first set of decoded dots and their corresponding indicies
    return genes_locations, dot_idx_filtered

def return_highly_expressed_names(decoded):
    """
    Returns list of top 5% of highly expressed genes
    Parameters
    -----------
    decoded = decoded genes csv file
    """
    
    #collapse into gene counts
    counts_df = Counter(decoded["genes"])
    #change to df
    counts_df = pd.DataFrame.from_dict(counts_df, orient='index').reset_index()
    #remove fakes
    counts_df_false = counts_df[counts_df["index"].str.startswith('fake')]
    counts_df_true = counts_df.drop(counts_df_false.index).reset_index(drop=True)
    #sort and identify top 5% highly expressed genes
    counts_df_true = counts_df_true.sort_values(0, ascending=False).reset_index(drop=True)
    highexpgenes = counts_df_true["index"][:int((len(counts_df_true)*0.05))].to_list()
    
    return highexpgenes
    
def dash_radial_decoding(location_path, codebook_path,
                         num_barcodes = 4, first_radius=1, second_radius=1,third_radius=1,diff=0,
                         min_seed=4, high_exp_seed=1, hybs = 12, output_dir = "", 
                         include_undecoded = False, decode_high_exp_genes = True,
                         triple_decode=True, decode_across = False):
    """
    This function will perform radial decoding on all barcodes as reference. Dot sequences
    that appear n number of times defined by min seed will be kept. Additionally, this function will run
    radial_decoding parallel three times. The first pass is to decode most of the crowded dots, the second pass
    will be with those dots removed, and the third will be with the dots removed from second round. Each round will recover 
    significantly less than the previous round. Three rounds should maximize recovery. 
    
    Parameters 
    ----------
    location_path = path to location.csv
    codebook_path = path to codebook showing at which channel and hyb a dot should appear
    num_barcodes = number of total barcodes
    first_radius = first search radius in pixels
    second_radius = second search radius in pixels
    third_radius = third search radius in pixels
    diff = allowed pseudocolor drops
    min_seed = number of times the same pseudocolor appears with changing reference
    high_exp_seed = number of min seeds to identify highly expressed genes
    hybs = total number of hybs
    output_dir = directory to where you want the file outputted
    include_undecoded = bool to output the undecoded dots
    decode_high_exp_genes = decode highly expressed genes first
    triple_decode=bool to perform another around of decoding
    decode_across = bool for decoding across
    
    Returns
    --------
    gene_locations.csv
    """
    #read in flocations file and drop unnamed columns
    locations = pd.read_csv(location_path)
    unwanted_columns = []
    for names in locations.columns:
        if "Unnamed" in names:
            unwanted_columns.append(names)
    locations = locations.drop(unwanted_columns, axis=1)
    #read in codebook
    codebook = pd.read_csv(codebook_path)
    codebook = codebook.set_index(codebook.columns[0])
    location_path_name = Path(location_path).name
    #collect z info
    z_info = location_path_name.split("_")[2].replace(".csv","")
    
    #check to see if the necessary amount of hybs are present
    assert len(locations["hyb"].unique()) >= hybs, "Locations file is missing a hyb"

    #make directories
    Path(output_dir).mkdir(parents=True, exist_ok = True)
    output_path = Path(output_dir) / f"diff_{diff}_minseed_{min_seed}_z_{z_info}_finalgenes.csv"
    
    #only keep top 5% of highly expressed genes
    if decode_high_exp_genes == True:
        #run decoding first pass
        decoded_1, indicies_used_1 = radial_decoding_parallel(locations, codebook,
                    num_barcodes=num_barcodes, radius=first_radius, diff=diff,
                    min_seed=high_exp_seed, hybs = hybs, include_undecoded = False, decode_across=decode_across)
        #get highly expressed genes
        highexpgenes = return_highly_expressed_names(decoded_1)
        #initialize loop
        highexpgenes_2 = highexpgenes.copy()
        locations_temp = locations.copy()
        #record of highly expressed decoded genes
        record_genes = []
        record_dots_used = []
        while set(highexpgenes) & set(highexpgenes_2) != set():
            #only used ones that overlap
            highexpgenes_overlap = list(set(highexpgenes) & set(highexpgenes_2))
            #only take out highly expressed genes
            decoded_1 = decoded_1.loc[decoded_1["genes"].isin(highexpgenes_overlap)]
            record_genes.append(decoded_1)
            indicies_used_df = decoded_1.index.tolist()
            #isolate used indicies in dot indicies list
            indicies_used_1 = list(map(indicies_used_1.__getitem__, indicies_used_df))
            #flatten 1st set of indicies used list
            flattened_indicies_used = [element for sublist in indicies_used_1 for element in sublist]
            #record used dots
            record_dots_used.append(locations_temp.iloc[flattened_indicies_used])
            #remove already decoded dots
            locations_temp = locations_temp.drop(flattened_indicies_used).reset_index(drop=True)
            #run decoding with 0.75 pixel search
            decoded_1, indicies_used_1 = radial_decoding_parallel(locations_temp, codebook,
                                                                  num_barcodes=num_barcodes, radius=first_radius,diff=diff,
                                                                  min_seed=high_exp_seed, hybs = hybs, 
                                                                  include_undecoded = False, 
                                                                  decode_across=decode_across)
            #get new highly expressed gene list
            highexpgenes_2 = return_highly_expressed_names(decoded_1)
        #remove last item in record dots used after exitting loop
        del record_dots_used[-1]
        #combine final highly expressed genes
        decoded_1 = pd.concat(record_genes).reset_index(drop=True)
        #use locations temp after exitting loop for second round
        new_locations = locations_temp.copy()
        del locations_temp
        
    else:
        #run decoding first pass
        decoded_1, indicies_used_1 = radial_decoding_parallel(locations, codebook,
                    num_barcodes=num_barcodes, radius=first_radius,diff=diff,
                    min_seed=min_seed, hybs = hybs, include_undecoded = False, decode_across=decode_across)
        #only get the indicies of decoded genes (excluding undefined)
        #the index on decoded will correspond to the same index in indicies used
        indicies_used_df = decoded_1.index.tolist()
        #isolate used indicies in dot indicies list
        indicies_used_1 = list(map(indicies_used_1.__getitem__, indicies_used_df))
        #flatten 1st set of indicies used list
        flattened_indicies_used = [element for sublist in indicies_used_1 for element in sublist]
        #remove already decoded dots
        new_locations = locations.drop(flattened_indicies_used).reset_index(drop=True)
        
    #output results from first pass
    decoded_1.sort_values("genes").to_csv(str(output_path).replace("finalgenes","round1"))
    
    #run decoding second pass with same or different search radius
    decoded_2, indicies_used_2 = radial_decoding_parallel(new_locations, codebook,
                                                          num_barcodes=num_barcodes, radius=second_radius,diff=diff,
                                                          min_seed=min_seed, hybs = hybs, include_undecoded = include_undecoded, 
                                                          decode_across=decode_across)
    if triple_decode == True:
        #output results from second pass
        decoded_combined = pd.concat([decoded_1, decoded_2])
        decoded_combined.sort_values("genes").reset_index(drop=True).to_csv(str(output_path).replace("finalgenes","round2"))
        #remove undefineds and get index of only decoded genes
        decoded_2 =  decoded_2[decoded_2["genes"] != "Undefined"]
        indicies_used_df = decoded_2.index.tolist()
        #isolate used indicies in dot indicies list
        indicies_used_2 = list(map(indicies_used_2.__getitem__, indicies_used_df))
        
        #flatten 2nd set of indicies used list
        flattened_indicies_used_2 = [element for sublist in indicies_used_2 for element in sublist]

        #remove already decoded dots
        new_locations_2 = new_locations.drop(flattened_indicies_used_2).reset_index(drop=True)

        #run decoding third pass
        try:
            decoded_3, indicies_used_3 = radial_decoding_parallel(new_locations_2, codebook,
                        num_barcodes=num_barcodes, radius=third_radius,diff=diff,
                        min_seed=min_seed, hybs = hybs, include_undecoded = include_undecoded, decode_across=decode_across)
            #combine decoded dfs
            decoded_combined = pd.concat([decoded_1, decoded_2, decoded_3])
        except:
            with open("decoding_message.txt", "w+") as f:
                f.write("Triple decoding did not yield anything.")
                f.close()
            decoded_combined = pd.concat([decoded_1, decoded_2])   
    else:
        #combine decoded dfs
        decoded_combined = pd.concat([decoded_1, decoded_2])
    
    #sort and reset index
    final_decoded = decoded_combined.sort_values("genes").reset_index(drop=True)
    
    #write files
    final_decoded.to_csv(str(output_path))
    
    #output used and unused dot locations
    #---------------------------------------------------------------------------------------------
    #collection dots used in round 1 decoding
    if decode_high_exp_genes == True:
        locations1 = pd.concat(record_dots_used).reset_index(drop=True)
    else:
        locations1 = locations.iloc[flattened_indicies_used]
        
    #collect dots used in second round
    if triple_decode == False:
        #flatten 2nd set of indicies used list
        flattened_indicies_used_2 = [element for sublist in indicies_used_2 for element in sublist]
    locations2 = new_locations.iloc[flattened_indicies_used_2]
    
    if triple_decode == True:
        #in case there was no output from triple decode
        try:
            #flatten 3rd set of indicies used list
            flattened_indicies_used_3 = [element for sublist in indicies_used_3 for element in sublist]
            locations3 = new_locations_2.iloc[flattened_indicies_used_3]
            #combine used dots
            dots_used = pd.concat([locations1,locations2,locations3]).reset_index(drop=True)
            #remove used dots
            dots_unused = new_locations_2.drop(flattened_indicies_used_3).reset_index(drop=True)
            #write out used dots
            dots_used.to_csv(str(output_path.parent / f"dots_used_z_{z_info}.csv"))
            #write out unused dots
            dots_unused.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
        except:
            dots_used = pd.concat([locations1,locations2]).reset_index(drop=True)
             #write out used dots
            dots_used.to_csv(str(output_path.parent / f"dots_used_z_{z_info}.csv"))   
            #write out unused dots
            new_locations_2.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
    else:
        #remove already decoded dots
        dots_unused= new_locations.drop(flattened_indicies_used_2).reset_index(drop=True)
        #combined dots used
        dots_used = pd.concat([locations1,locations2]).reset_index(drop=True)
        #write out used dots
        dots_used.to_csv(str(output_path.parent / f"dots_used_z_{z_info}.csv"))
        #write out unused dots
        dots_unused.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
        
    #Output percent decoded
    percent_decoded = str(round(((len(dots_used)/len(locations)) * 100),2))
    percent_output = output_path.parent / f"percent_decoded_z_{z_info}.txt"
    with open(str(percent_output),"w+") as f:
        f.write(f'Percent of dots decoded = {percent_decoded}')
        f.close()
    
    
                         
