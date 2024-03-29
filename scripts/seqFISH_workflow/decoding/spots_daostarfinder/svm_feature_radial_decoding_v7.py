"""
author: Katsuya Lex Colon
updated: 12/04/2023
"""

#general analysis packages
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from itertools import product
import time
#parallel processing
from concurrent.futures import ProcessPoolExecutor
#path management
from pathlib import Path
import sys
import os
# SVM model
from sklearn.svm import SVC
#for splitting training and test set
from sklearn.model_selection import train_test_split
#screening hyperparameters
from sklearn.model_selection import GridSearchCV
#scaling features
from sklearn.preprocessing import MinMaxScaler
#metrics module for model accuracy calculation
from sklearn import metrics
#output most important features
from sklearn.inspection import permutation_importance
#plotting package
import matplotlib.pyplot as plt
import seaborn as sns

#ignore general warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 
    
class decode:
    
    def __init__(self, location_path, codebook_path, num_rounds,first_radius,
                 second_radius, third_radius, diff, min_seed, high_exp_seed, total_hybs,
                 probability_cutoff, desired_fdr,parity_round, include_undefined,
                 decode_high_exp_genes_first, triple_decode, score_brightness, blank_ch, output_dir):
        """
        Parameters 
        ----------
        location_path: path to location.csv
        codebook_path: path to codebook showing at which channel and hyb a dot should appear
        num_rounds: number of barcoding rounds
        first_radius: first search radius in pixels
        second_radius: second search radius in pixels
        third_radius: third search radius in pixels
        diff: allowed pseudocolor drops
        min_seed: number of times the same pseudocolor appears with changing reference
        high_exp_seed: number of min seeds to identify highly expressed genes
        total_hybs: total number of hybs
        probability_cutoff: probability threshold cutoff to remove low scoring dots before decoding
        desired_fdr: value of desired fdr
        parity_round: bool if you included parity round
        include_undefined: bool to output the undecoded dots
        decode_high_exp_genes_first: decode highly expressed genes first
        triple_decode: bool to perform another around of decoding
        score_brightness: bool to score brightness
        output_dir: directory to where you want the file outputted
        """
        
        self.location_path = location_path
        self.codebook_path = codebook_path
        self.num_barcodes = num_rounds
        self.first_radius = first_radius
        self.second_radius = second_radius
        self.third_radius = third_radius
        self.diff = diff
        self.min_seed = min_seed
        self.high_exp_seed = high_exp_seed
        self.hybs = total_hybs
        self.probability_cutoff = probability_cutoff
        self.desired_fdr = desired_fdr
        self.parity_round = parity_round
        self.include_undefined = include_undefined
        self.decode_high_exp_genes = decode_high_exp_genes_first
        self.triple_decode = triple_decode
        self.score_brightness = score_brightness
        self.blank_ch = blank_ch
        self.output_dir = output_dir
        
    
    def poly_rbf_gen_dot_probabilities(dots_used_trues, dots_used_fakes, locations):
        """
        A SVM classifier using a radial basis function or polynomial kernel to assign probability scores to dots.
    
        Parameters
        ----------
        dots_used_trues: dot locations of true dots with features to train svm classifier
        dots_used_fakes: dot locations of fake dots with features to train svm classifier
        locations: all of the dots

        Returns
        -------
        true_probs_only: probability of being true for each spot
        X_test: test points
        y_test: test labels
        locations: dot locations with probabilities
        plt: feature importance plot
        """
        #get attributes
        trues = dots_used_trues[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
        trues["labels"] = 1
        fakes = dots_used_fakes[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
        fakes["labels"] = -1
    
        #make sure number of fakes is > 500
        assert len(fakes) >= 500, "Not enough fake dots to train."
    
        #if there is over 500,000 fake spots then down sample for training
        fakes = fakes.iloc[:500000,:]
    
        #downsample trues to match fakes
        #generally there will be less fakes then trues 
        trues = trues.iloc[:len(fakes),:]
    
        #combine trues and fakes into one array
        comb = pd.concat([trues,fakes]).reset_index(drop=True)
    
        #Split dataset into training set and test set
        # 80% training and 20% test
        X_train, X_test, y_train, y_test = train_test_split(comb.iloc[:,0:6], 
                                                            comb["labels"], test_size=0.20,random_state=42, 
                                                            stratify=comb["labels"])
    
        # Train classifiers
        start = time.time()
    
        #Normalize features
        scaler = MinMaxScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
    
        #parameters to test
        C_range = np.logspace(1, 12, 12)
        gamma_range = np.logspace(-9, 1, 12)
        degree_range = [1,2,3,4,5]
        param_grid_poly = dict(degree=degree_range, C=C_range)
        param_grid_rbf = dict(gamma=gamma_range, C=C_range)
    
        #total number of cross validations
        total_cross_val = 8
    
        #total number of fits
        total_fits = len(C_range) * len(degree_range) * total_cross_val
    
        print(f"Performing hyperparameter optimization using polynomial kernel totalling {total_fits} fits...")
    
        #screen all permutation of parameters
        grid_poly = GridSearchCV(SVC(kernel="poly", max_iter=2e5, coef0=1, gamma=1), 
                            param_grid=param_grid_poly, cv=total_cross_val,  n_jobs=-1,)
        grid_poly.fit(X_scaled, y_train)
    
        #which is the best
        print("The best parameters are %s with a score of %0.2f" % (grid_poly.best_params_, grid_poly.best_score_))
        print(f"Model building took {round((time.time()-start)/60, 2)} min and is now being applied to test set.")
        #generate probability model based on best params
        clf = SVC(kernel="poly", max_iter=1e6, coef0=1,gamma=1,
                  degree=grid_poly.best_params_["degree"], 
                  C=grid_poly.best_params_["C"], probability=True)
        clf.fit(X_scaled, y_train)
        # Model Accuracy: how often is the classifier correct with its labels
        #normalize data
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        #obtain probabilities
        y_proba = clf.predict_proba(X_test_scaled)
        true_probs_only_poly = y_proba[:,1]
        print(f"Label classification accuracy: {round(metrics.accuracy_score(y_test, y_pred), 2)}")
        poly_accuracy = metrics.accuracy_score(y_test, y_pred)
        
        #total number of fits
        total_fits = len(C_range) * len(gamma_range) * total_cross_val
    
        print(f"Performing hyperparameter optimization usinf rbf kernel totalling {total_fits} fits...")
    
        #screen all permutation of parameters
        grid_rbf = GridSearchCV(SVC(kernel="rbf", max_iter=2e5, coef0=1), 
                            param_grid=param_grid_rbf, cv=total_cross_val,  n_jobs=-1,)
        grid_rbf.fit(X_scaled, y_train)
    
        #which is the best
        print("The best parameters are %s with a score of %0.2f" % (grid_rbf.best_params_, grid_rbf.best_score_))
        print(f"Model building took {round((time.time()-start)/60, 2)} min and is now being applied to test set.")
        #generate probability model based on best params
        clf = SVC(kernel="rbf", max_iter=1e6, 
                  coef0=1,gamma = grid_rbf.best_params_["gamma"], 
                  C=grid_rbf.best_params_["C"], probability=True)
        clf.fit(X_scaled, y_train)
        #normalize data
        X_test_scaled = scaler.transform(X_test)
        y_pred = clf.predict(X_test_scaled)
        #obtain probabilities
        y_proba = clf.predict_proba(X_test_scaled)
        true_probs_only_rbf = y_proba[:,1]
        print(f"Label classification accuracy: {round(metrics.accuracy_score(y_test, y_pred), 2)}")
        rbf_accuracy = metrics.accuracy_score(y_test, y_pred)
        
        #pick model with better kernel 
        best = np.argmax([poly_accuracy, rbf_accuracy])
        
        #create final model
        if best == 0:
            clf = SVC(kernel="poly", max_iter=1e6, coef0=1,gamma=1,
                  degree=grid_poly.best_params_["degree"], 
                  C=grid_poly.best_params_["C"], probability=True)
            clf.fit(X_scaled, y_train)
            true_probs_only = true_probs_only_poly
            print("Poly kernel was picked...")
        else:
            clf = SVC(kernel="rbf", max_iter=1e6, 
                  coef0=1,gamma = grid_rbf.best_params_["gamma"], 
                  C=grid_rbf.best_params_["C"], probability=True)
            clf.fit(X_scaled, y_train)
            true_probs_only = true_probs_only_rbf
            print("RBF kernel was picked...")
        
        print("Generating probabilities for each dot...")
        #get probabilities for every dot
        locations_features = locations[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
        #normalize data
        locations_scaled = scaler.transform(locations_features)
        loc_proba = clf.predict_proba(locations_scaled)
        locations["probability on"] = loc_proba[:,1]
        print(f"Probabilities assigned, Total time = {round((time.time()-start)/60, 2)} min")
    
        #output most important features
        perm_importance = permutation_importance(clf, X_test_scaled, y_test)
        features= X_test.columns
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Feature Importance")

        return true_probs_only, X_test, y_test, locations, plt

    def false_positive_rate(gene_locations, truebook, fakebook):
        """
        Calculate false positive rate.

        Parameters
        ----------
        gene_locations: decoded gene locations
        truebook: codebook of only real genes
        fakebook: codebook of fake genes

        Returns
        -------
        fpr: false positive rate
        """
        #make fake barcodes df
        fakebrcds = gene_locations[gene_locations["genes"].str.startswith("fake")]
        #make real barcodes df
        real = gene_locations.drop(fakebrcds.index, axis=0)
        
        M_on = len(truebook) #number of true barcodes in codebook
        M_off = len(fakebook) #number of fake barcodes in codebook
        N_on = len(real) #number of observed true barcodes
        N_off = len(fakebrcds)#number of observed fake barcodes
        false_count_freq = N_off/M_off #frequency of false barcodes
        false_positive_counts = M_on*false_count_freq #estimated false positive barcodes based on frequency of false codes
        fpr = false_positive_counts/N_on  #normalized false positive rate

        return fpr

    def set_fpr(gene_locations, codebook, fpr_cutoff=0.05):
        """
        Function to sort barcodes by codeword score and sample subsets of decreasing codeword score
        while calculating false positive rate. A final gene locations file will be outputted based on user defined
        fpr_cutoff.
        
        Parameters
        ----------
        gene_locations: decoded gene locations file
        fpr_cutoff: desired fdr value

        Returns
        -------
        desired_df: fpr adjusted dataframe
        plt: fpr plot
        """
        #sort barcodes by codeword score
        sorted_genes = gene_locations.sort_values("codeword score", ascending=False).reset_index(drop=True)
        
        #separate true and fake codebook
        fakebook = codebook[codebook.index.str.startswith("fake")]
        truebook = codebook.drop(fakebook.index)
        
        #calculate fpr while sampling barcodes
        collect_fpr = []
        for i in np.linspace(100, len(sorted_genes),1000).astype(int):
            iso_loc = sorted_genes.iloc[0:i]
            #do not include fakes generated from two separate masks
            if "cell number" in iso_loc.columns:
                iso_loc = iso_loc[iso_loc["cell number"].astype(int) == iso_loc["cell number"]].reset_index(drop=True)
            #calculate fdr
            fpr_val = decode.false_positive_rate(iso_loc, truebook, fakebook)
            collect_fpr.append([i,fpr_val])  
         
        #convert to array
        collect_fpr = np.array(collect_fpr)
        
        #get fpr below cutoff with the highest number of decoded barcodes
        filtered_by_fpr = collect_fpr[np.where(collect_fpr[:,1]<=fpr_cutoff)[0]]
        #in case nothing is below cutoff then return empty df
        if filtered_by_fpr.size == 0:
            desired_df = pd.DataFrame()
            #generate plot
            plt.plot(collect_fpr[:,0],collect_fpr[:,1])
            plt.xlabel("Decoded Barcodes")
            plt.ylabel("FPR")
            if fpr_cutoff != None:
                plt.axhline(fpr_cutoff, c="red", ls = "--", label="FPR cutoff")
            plt.legend()
            sns.despine()
        else:
            max_dots = filtered_by_fpr[np.argmax(filtered_by_fpr[:,0])]
            desired_df = sorted_genes.iloc[0:int(max_dots[0])]
            #generate plot
            plt.plot(collect_fpr[:,0],collect_fpr[:,1])
            plt.xlabel("Decoded Barcodes")
            plt.ylabel("FPR")
            if fpr_cutoff != None:
                plt.axhline(fpr_cutoff, c="red", ls = "--", label="FDR cutoff")
            plt.scatter(max_dots[0],max_dots[1], color = "k", label="Best")
            plt.legend()
            sns.despine()
        
        return desired_df,plt

    def filter_dots_fast(self):
        """
        Keep unique sets that appear >= min_seed times.
        
        Parameters
        ----------
        dots_idx: list of dot sets as tuples
        min_seed: number of times a dot sequence should appear
        
        Returns
        --------
        filtered_set: dots that appeared >= min_seed times
        """
        
        #generate frozen set
        flat_list = [frozenset(element) for element in self.dots_index]
        #count the number of times a frozen set appeared
        obj = Counter(flat_list)
        #convert to df
        df = pd.DataFrame.from_dict(obj, orient='index').reset_index()
        #keep sets that appear >= the min seed
        filtered_set = df[df[0] >= self.min_seed]["index"].to_list()
        
        #return unique sets
        return filtered_set
    
    def score_codewords(code_list,codeword_hash, distance_hash):
        """
        This function will pick best overall codeword.
        
        Parameters
        ----------
        code_list: list of codword sets we are trying to compare
        codeword_hash: a hash table to return overall codeword score
        distance_hash: a hash table to return overall distance for codeword
        
        Returns
        ---------
        best_code: best scoring codeword
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
        from score_codewords() function is not repeated in the other choices.
        
        Parameters
        ----------
        code_list: list of codeword sets we analyzed from score_codewords()
        best_codeword: the best codeword set chosen from score_codewords()
        
        Returns
        ----------
        recovered_codes: codewords that were recovered if the dots are different from the best_codeword
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
        """
        This function will try to pick between multiple codewords based on codeword
        score generated from radial_decoding function and overall codeword distance.
        
        Parameters
        ----------
        filtered_set: ouput from filter_dots_fast
        codeword_scores: output from radial decoding
        total_distances: output from radial decoding
        
        Returns
        ----------
        final_codewords: final set of dots that are unique and high scoring
        """
        
        #group all sets with any matching dot indicies using individual unique dot sets as queries
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
    
        #group again to combine list of dot sets that have any common dot sequence
        new_set = []
        #keep record of which indicies were combined
        history_of_comb = []
        #keep record of which indicies have been used
        history_index = []
        for i in range(len(matching_element_list)):
            #create copy of list of sets
            comb_set = matching_element_list[i].copy()
            #check if any dot set overlaps with any other list of sets
            for dot_set in matching_element_list[i]:
                for j in range(len(matching_element_list)):
                    #if dot set overlaps with another list of sets combine them
                    if (dot_set in matching_element_list[j]) and (i!=j) and (set([i,j]) not in history_of_comb):
                        comb_set += matching_element_list[j]
                        history_of_comb.append(set([i,j]))
                        history_index += [i,j]
            #check to see if anything was combined and if index was used already
            if (len(comb_set) == len(matching_element_list[i])) and (i not in history_index):
                new_set.append(matching_element_list[i])
            elif (len(comb_set) == len(matching_element_list[i])) and (i in history_index):
                continue
            else:
                new_set.append(list(set(comb_set)))
    
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
        #store best dot sequence
        #there is a small probability that a specific dot set could still end up in two separate lists, this step will account for that
        final_history = set()
        for codes in new_set:
            #get best code word from list of sets
            best_code = decode.score_codewords(codes,codeword_hash, distance_hash)
            #check if best code was already chosen
            while best_code & final_history:
                codes.remove(best_code)
                try:
                    best_code = decode.score_codewords(codes,codeword_hash, distance_hash)
                except ValueError:
                    best_code = 0
                    break
            if best_code == 0:
                continue
            #keep record
            for dot_idx in best_code:
                final_history.add(dot_idx)
            final_codewords.append(list(best_code))
            #check to see if we can recover any codewords
            recovered_codes = decode.recover_codes(codes, best_code)
            #if there is one returned and it has not been picked before we will add to final
            if (len(recovered_codes) == 1) and (recovered_codes[0] & final_history == set()):
                final_codewords.append(list(recovered_codes[0]))
                #keep record
                for dot_idx in recovered_codes[0]:
                    final_history.add(dot_idx)
            #if nothing is returned then we will continue
            elif len(recovered_codes) == 0:
                continue
            #if there is multiple choices we will perform a while loop until the choices become 0
            else:
                while len(recovered_codes) != 0:
                    #find the best codeword in recovered list
                    best_2 = decode.score_codewords(recovered_codes,codeword_hash, distance_hash)
                    while best_2 & final_history:
                        recovered_codes.remove(best_2)
                        if (len(recovered_codes) == 1) and (recovered_codes[0] & final_history == set()):
                            best_2 = recovered_codes[0]
                            break
                        elif len(recovered_codes) == 0:
                            break
                        else:
                            best_2 = decode.score_codewords(recovered_codes,codeword_hash, distance_hash)
                    if len(recovered_codes) == 0:
                        continue
                    else:
                        final_codewords.append(list(best_2))
                    #keep record
                    for dot_idx in best_2:
                        final_history.add(dot_idx)
                    #check to see if we can recover any codewords again
                    recovered_2 = decode.recover_codes(recovered_codes, best_2)
                    #if there is one choice returned we can add
                    #otherwise we will repeat loop or end loop
                    if (len(recovered_2) == 1) and (recovered_2[0] & final_history == set()):
                        final_codewords.append(list(recovered_2[0]))
                        #keep record
                        for dot_idx in recovered_2[0]:
                            final_history.add(dot_idx)
                        break
                    elif (len(recovered_2) == 1) and (recovered_2[0] & final_history):
                        break
                    #reassign recovered_codes
                    recovered_codes = recovered_2.copy()
    
        #delete some variables
        del matching_element_list
        del new_set
        del search_set
        del history_element
        del codeword_hash
        del distance_hash
        
        return final_codewords
    
    def find_best_spot_combination(temp, tempscore, locations, neighbor_list, distance_list,
                                   hybs=24, num_barcodes=4, blank_ch = False):
        """
        Generates all possible codewords from highest scoring to lowest scoring and 
        pick highest scoring codeword that passes parity.
        
        Parameters
        ----------
        temp: temporary list containing possible spot indicies
        tempscore: temporary list containing scores for each spot
        locations: the spot locations file
        neighbor_list: list of dot indicies that are neighbors
        distance_list: euclidean distance of each spot to seed
        hybs: total number of hybs
        num_barcodes: total number of barcoding rounds
        blank_ch: bool on whether there are empty channels on certain hybs

        Returns
        -------
        list(best): best set of spots
        best_score: score associated with best set
        best_dist: total distance associated with best set
        """
        
        #generate all possible codeword combination
        possible_codes = list(product(*temp, repeat=1))
        
        #make spot score dictionary
        temp_flat = [element for sublist in temp for element in sublist]
        tempscore_flat = [element for sublist in tempscore for element in sublist]
        score_dict = dict(zip(temp_flat, tempscore_flat))
        
        #generate total distance for all possible codes
        possible_code_distances = []
        for code in possible_codes:
            total_distance_values=0
            for k,dot_idx in enumerate(code):
                if k != 0:
                    dist_idx = neighbor_list[k].index(dot_idx)
                    total_distance_values += distance_list[k-1][dist_idx]
            possible_code_distances.append(total_distance_values)
        
        #calculate overall score for each possible codeword
        scores = []
        for codes in possible_codes:
            scores.append(np.sum(list(map(score_dict.get, codes))))
        
        #find scores that are the same from highest to lowest
        unique_scores = sorted(list(set(scores)))[::-1]
        index_match = []
        for score in unique_scores:
            indices = [i for i, x in enumerate(scores) if x == score]
            index_match.append(indices)
        
        #sort the matching score indicies by shortest overall distance
        new_sorted_indicies = []
        for index in index_match:
            dist = [possible_code_distances[i] for i in index]
            dist_sort = np.argsort(dist)
            reorg_index = [index[i] for i in dist_sort]
            new_sorted_indicies += reorg_index
            
        #sort possible codes by score and overall distance
        sorted_scores = [scores[index] for index in new_sorted_indicies]
        sorted_codes = [possible_codes[index] for index in new_sorted_indicies]
        sorted_dist = [possible_code_distances[index] for index in new_sorted_indicies]
        
        #pseudocolor calculation will be dependent on whether it is across or within
        if len(locations.ch.unique()) == 1:
            #check each codeword on whether it passes parity 
            for i,codes in enumerate(sorted_codes):
                potential = (locations.iloc[list(codes)].sort_values("hyb")["hyb"].values) + 1
                pseudocolors = int(hybs/num_barcodes)
                #convert into pseudocolor sequence
                get_barcode = []
                for k,ps in enumerate(potential):
                    get_barcode.append(ps - (k*pseudocolors))
                #if a barcode is missing a pseudocolor then we can't do parity check
                if len(get_barcode)==(num_barcodes-1):
                    continue
                #if the codeword passes parity then end loop
                parity_code = sum(get_barcode[:len(get_barcode)-1]) % pseudocolors
                if parity_code == 0:
                    parity_code = pseudocolors
                if parity_code == get_barcode[-1]:
                    best = codes
                    best_score = sorted_scores[i]
                    best_dist = sorted_dist[i]
                    break
    
            #if none passed parity then just return first code since it is the best scoring
            if "best" not in locals():
                best = sorted_codes[0]
                best_score = sorted_scores[0]
                best_dist = sorted_dist[0]
    
            return list(best), best_score, best_dist
        else:
            #figure out total pseudocolors
            hybs_per_round = int(hybs/num_barcodes)
            unique_ch = len(locations.ch.unique())
            pseudocolors = hybs_per_round * unique_ch
            #only run this if blank channels are present
            if blank_ch == True:
                round1_last_hyb = locations[locations.hyb == (hybs_per_round-1)]
                #counts per ch
                ch_counts = []
                for ch in range(1,unique_ch+1,1):
                    ch_counts.append(len(round1_last_hyb[round1_last_hyb.ch==ch]))
                #if a certain set of channels only have less than 10% of max spots then assume it is blank
                threshold = max(ch_counts)*0.1
                blank_ch = np.sum(np.array(ch_counts)<threshold)
                pseudocolors = int(pseudocolors-blank_ch)
            #check each codeword on whether it passes parity 
            for i,codes in enumerate(sorted_codes):
                potential = locations.iloc[list(codes)].sort_values("hyb")[["hyb","ch"]]
                potential = potential.values
                #convert into pseudocolor sequence
                get_barcode = []    
                for k,ps in enumerate(potential):
                    get_barcode.append(((ps[0])*3)+ps[1]-(k*pseudocolors)) 
                #if a barcode is missing a pseudocolor then we can't do parity check
                if len(get_barcode)==(num_barcodes-1):
                    continue
                #if the codeword passes parity then end loop
                parity_code = sum(get_barcode[:len(get_barcode)-1]) % pseudocolors
                if parity_code == 0:
                    parity_code = pseudocolors
                if parity_code == get_barcode[-1]:
                    best = codes
                    best_score = sorted_scores[i]
                    best_dist = sorted_dist[i]
                    break
    
            #if none passed parity then just return first code since it is the best scoring
            if "best" not in locals():
                best = sorted_codes[0]
                best_score = sorted_scores[0]
                best_dist = sorted_dist[0]
    
            return list(best), best_score, best_dist
        
    def radial_decoding(self,locations,num_barcodes = 4, 
                        radius=np.sqrt(2),diff=0, cycle=1,
                        seed=0, hybs = 12, look_for_parity=False):
        """
        This function will decode dots utilizing kNN algorithm from sklearn using the euclidean formula
        as a measure of distance. Essentially, a defined search radius will be used to identify nearby dots. 
        Dot sequence will be determined using a score-based metric incorporating distance, intensity, and size. 
        Dot sequences that appear n number of time defined by min seed will be kept. A codeword score will also
        be assigned based on ambiguity (total number of neighbors for each ref). A table will be generated for each unnamed
        gene, at which a codebook will act as a hash table for identification. 
        
        Parameters
        ----------
        locations: locations.csv file
        num_barcodes: number of readout sites
        radius: search radius using euclidean metric
        diff: allowed barcode drops
        seed: which barcode set to use as reference
        hybs: total number of hybs
        look_for_parity: look for codes that have parity 
        
        Returns
        --------
        dot_idx: best picked dot indicies 
        ambiguity_scores: sum of all neighbors for each unique set of dot indicies
        codeword_score_list: score for each possible codeword
        total_distance_list: total distance for each possible codeword
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
                if hybs==num_barcodes:
                    barcode = locations[locations["hyb"] == h]
                    barcoding_round.append(temp[0])
                    barcoding_round.append(barcode)
                else:
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
        
        if look_for_parity == True:
            dot_idx = []
            ambiguity_scores = []
            codeword_score_list = []
            total_distance_list = []
            for i in range(len(neighbor_list2)):
                temp = []
                tempscore = []
                #calculate overall scores
                ambiguity = 0
                codeword_score = 0
                total_distance_values = 0
                for j in range(len(neighbor_list2[i])):
                    if len(neighbor_list2[i][j]) == 1:
                        #add single spot with no neigbors
                        temp.append([neighbor_list2[i][j][0]])
                        #add max score possible for single spot with no neighbbors
                        #also add ambiguity score of 1 (counting self)
                        if self.score_brightness == True:
                            tempscore.append([2])
                        else:
                            tempscore.append([1])
                        ambiguity += len(neighbor_list2[i][j])
                    else:
                        #generate scoring system
                        ###previous was 2,1,0.5 (dist-57%,int-29%,size-14%); current (dist-50%,int-37.5%,size-12.5%)
                        dist_values = np.linspace(0,1,len(neighbor_list2[i][j])+1)[::-1]
                        if self.score_brightness == True:
                            int_values = np.linspace(0,0.75,len(neighbor_list2[i][j])+1)[::-1]
                            size_values = np.linspace(0,0.25,len(neighbor_list2[i][j])+1)[::-1]
                        else:
                            #for across channel lantern
                            int_values = np.zeros(len(neighbor_list2[i][j]))
                            size_values = np.zeros(len(neighbor_list2[i][j]))
                        #get dot traits
                        #note that pandas iloc preserve order matching slice order
                        dot_traits = locations.iloc[neighbor_list2[i][j]]
                        #generate score table
                        trait_score = np.zeros(len(dot_traits))
                        #rank dots 
                        int_score = np.argsort(dot_traits["flux"]).values[::-1]
                        size_score = np.argsort(dot_traits["size"]).values[::-1]
                        #calculate best score
                        #note that distance is already sorted
                        for _ in range(len(trait_score)):
                            trait_score[int_score[_]] += int_values[_]
                            trait_score[_] += dist_values[_]
                            trait_score[size_score[_]] += size_values[_]
                        #score for each spot
                        temp.append(neighbor_list2[i][j])
                        tempscore.append(trait_score)
                        ambiguity += len(neighbor_list2[i][j])
                        
                #check if the selected spots pass parity and return best spot sequence
                best, best_score, best_dist = decode.find_best_spot_combination(temp, tempscore, locations, 
                                                                         neighbor_list2[i], 
                                                                         distance_list2[i],
                                                                         hybs=hybs, num_barcodes=num_barcodes, 
                                                                         blank_ch = self.blank_ch)
               
                #add codeword score and total distance
                codeword_score += best_score
                total_distance_values += best_dist
                
                #adjust final ambiguity score by total number of dots in set
                ambiguity_score_final = ambiguity-len(neighbor_list2[i])
                #adjust final codeword score by ambiguity+1 (add 1 to prevent division by 0)
                if cycle == 1:
                    codeword_score = codeword_score/((ambiguity_score_final+1)*0.10)
                elif cycle == 2:
                    codeword_score = codeword_score/((ambiguity_score_final+1)*0.50)
                else:
                    codeword_score = codeword_score/(ambiguity_score_final+1)
                #normalize codeword score by max dot score * number of barcode sites
                codeword_score = codeword_score/(2*num_barcodes)
                #append all of the data to final list
                dot_idx.append(best)
                ambiguity_scores.append([tuple(best),ambiguity_score_final])
                codeword_score_list.append([tuple(best),codeword_score])
                total_distance_list.append([tuple(best),total_distance_values])
    
            #no longer need this  
            del neighbor_list2
            del distance_list2
            del temp
        else:
            #Get best dot sequence based on dot traits
            #weighted scores will be calculated based on distance, flux, and size
            #probability adjusted or unadjusted overall codeword score will also be computed 
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
                        #add 1 for ambiguity but will become zero after adjustment in final step
                        ambiguity += 1
                        #max score a dot can get
                        if self.score_brightness == True:
                            codeword_score += 2.0
                        else:
                            codeword_score += 1.0 
                        if j == 0:
                            continue
                        else:
                            total_distance_values += distance_list2[i][j-1][0]
                    else:
                        #generate scoring system
                        ###previous was 2,1,0.5 (dist-57%,int-29%,size-14%); current (dist-50%,int-37.5%,size-12.5%)
                        dist_values = np.linspace(0,1,len(neighbor_list2[i][j])+1)[::-1]
                        if self.score_brightness == True:
                            int_values = np.linspace(0,0.75,len(neighbor_list2[i][j])+1)[::-1]
                            size_values = np.linspace(0,0.25,len(neighbor_list2[i][j])+1)[::-1]
                        else:
                            #for across channel lantern
                            int_values = np.zeros(len(neighbor_list2[i][j]))
                            size_values = np.zeros(len(neighbor_list2[i][j]))
                        #get dot traits
                        #note that pandas iloc preserve order matching slice order
                        dot_traits = locations.iloc[neighbor_list2[i][j]]
                        #generate score table
                        trait_score = np.zeros(len(dot_traits))
                        #rank dots and see if we are using flux or average intensity
                        int_score = np.argsort(dot_traits["flux"]).values[::-1]
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
    
                #adjust final ambiguity score by total number of dots in set
                ambiguity_score_final = ambiguity-len(neighbor_list2[i])
                #adjust final codeword score by ambiguity+1 (add 1 to prevent division by 0)
                if cycle == 1:
                    codeword_score = codeword_score/((ambiguity_score_final+1)*0.10)
                elif cycle == 2:
                    codeword_score = codeword_score/((ambiguity_score_final+1)*0.50)
                else:
                    codeword_score = codeword_score/(ambiguity_score_final+1)
                #normalize codeword score by max dot score * number of barcode sites
                codeword_score = codeword_score/(2*num_barcodes)
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
    
    def radial_decoding_parallel(self, locations, codebook, num_barcodes = 4, radius=1, diff=0, cycle=1,
                                 min_seed=4, hybs = 12, include_undecoded = False, 
                                 look_for_parity = False, parity_round=True):
        """This function will perform radial decoding on all barcodes as reference. Dot sequences
        that appear n number of times defined by min seed will be kept.
        Parameters 
        ----------
        locations: location.csv file
        codebook: codebook.csv
        num_barcodes: number of readout sites
        radius: search radius using euclidean metric
        diff: allowed barcode drops
        hybs: total number of hybs
        include_undecoded: bool to output the undecoded dots
        look_for_parity: bool to look for parity codes
        parity_round: bool on whether a parity round is included
        
        Returns
        --------
        genes_locations: gene locations df 
        dot_idx_filtered: dot indicies used
        """
        
        #make sure diff is not greater than 1
        assert diff < 2, "Diff cannot be > 1"
        
        #parallel processing for nearest neighbor computation for each barcode
        with ProcessPoolExecutor(max_workers=num_barcodes) as exe:
            futures = []
            for i in range(num_barcodes):
                seed = i
                fut = exe.submit(decode.radial_decoding, self, locations,
                                 num_barcodes, radius,diff,cycle,
                                 seed, hybs, look_for_parity)
                futures.append(fut)
                
        #collect result from futures objects
        result_list = [fut.result() for fut in futures]
        #split dot index list, ambiguity list, codeword score list and total distance while flattening them
        self.dots_index = [element for sublist in result_list for element in sublist[0]]
        ambiguity_list = [element for sublist in result_list for element in sublist[1]]
        codeword_score_list = [element for sublist in result_list for element in sublist[2]]
        total_distance_list = [element for sublist in result_list for element in sublist[3]]
        
        #keep dots that match min seed
        dot_idx_min = decode.filter_dots_fast(self)
            
        #pick best codewords where there could be multiple options
        dot_idx_filtered =  decode.pick_best_codeword(dot_idx_min, codeword_score_list, total_distance_list)
        
        #generate hash table for ambiguity score assignment
        amb_dict = {}
        for i in range(len(ambiguity_list)):
            #if the key exists, then skip
            if tuple(sorted(ambiguity_list[i][0])) in amb_dict:
                continue
            else:
                amb_dict.update({tuple(sorted(ambiguity_list[i][0])):ambiguity_list[i][1]})
                
        #generate hash table for codeword score assignment
        codeword_score_dict = {}
        for i in range(len(codeword_score_list)):
            #if the key exists, then skip
            if tuple(sorted(codeword_score_list[i][0])) in codeword_score_dict:
                continue
            else:
                codeword_score_dict.update({tuple(sorted(codeword_score_list[i][0])):codeword_score_list[i][1]})
    
        #make list of ambiguity scores for each dot sequence index
        ambiguity_scores_final =[]
        for i in range(len(dot_idx_filtered)):
            score = amb_dict[tuple(sorted(dot_idx_filtered[i]))]
            ambiguity_scores_final.append(score)
            
        #make list of codeword scores for each dot sequence index
        codeword_scores_final =[]
        for i in range(len(dot_idx_filtered)):
            score = codeword_score_dict[tuple(sorted(dot_idx_filtered[i]))]
            codeword_scores_final.append(score)
    
        #isolate dot info for barcode sets
        dot_info = []
        for idx in dot_idx_filtered:
            dot_info.append(locations.iloc[idx])
    
        #generate code table for decoding and store info about dots
        info_list = []
        code_table = np.zeros(shape=(len(dot_info), hybs)).astype(int)
        for i in range(len(dot_info)):
            code = dot_info[i][["hyb","ch"]].values
            if "cell number" in dot_info[i]:
                info = dot_info[i][["x","y","z","flux","max intensity","sharpness", "symmetry", "roundness by gaussian fits","size", "cell number"]].mean().values
            else:
                info = dot_info[i][["x","y","z","flux","max intensity","sharpness", "symmetry", "roundness by gaussian fits","size"]].mean().values 
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
                    if parity_round==False:
                        sys.exit("Check codebook for completeness")
                    else:
                        decoded_genes.append("Undefined")
    
            #add gene names
            genes_locations = pd.DataFrame(info_list)
            if len(genes_locations.columns) == 10:
                genes_locations.columns = ["x","y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits","size","cell number"]
            else:
                genes_locations.columns = ["x","y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits","size"]
            genes_locations["genes"] = decoded_genes
            #add ambiguity score
            genes_locations["ambiguity score"] = ambiguity_scores_final
            #add codeword score
            genes_locations["codeword score"] = codeword_scores_final
            if include_undecoded ==  False:
                genes_locations = genes_locations[genes_locations["genes"] != "Undefined"]
            if "cell number" in genes_locations:
                genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits", "size", "ambiguity score", "codeword score", "cell number"]] 
            else:
                genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits", "size", "ambiguity score", "codeword score"]] 
    
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
            if len(genes_locations.columns) == 10:
                genes_locations.columns = ["x","y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits","size", "cell number"]
            else:
                genes_locations.columns = ["x","y","z","brightness","peak intensity","sharpness", "symmetry", "roundness by gaussian fits","size"]
            genes_locations["genes"] = decoded_genes
            #add ambiguity score
            genes_locations["ambiguity score"] = ambiguity_scores_final
            #add codeword score
            genes_locations["codeword score"] = codeword_scores_final
            if include_undecoded ==  False:
                genes_locations = genes_locations[genes_locations["genes"] != "Undefined"]
            if "cell number" in genes_locations:
                genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity", "sharpness", "symmetry", "roundness by gaussian fits","size", "ambiguity score", "codeword score", "cell number"]]  
            else:
                genes_locations = genes_locations[["genes", "x", "y","z","brightness","peak intensity", "sharpness", "symmetry", "roundness by gaussian fits","size", "ambiguity score", "codeword score"]]  
            
        #return first set of decoded dots and their corresponding indicies
        return genes_locations, dot_idx_filtered
    
    def return_highly_expressed_names(decoded):
        """
        Returns list of top 10% of highly expressed genes
        
        Parameters
        -----------
        decoded: decoded genes csv file
        
        Returns
        --------
        highexpgenes: list of highly expressed gene names
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
        highexpgenes = counts_df_true["index"][:int((len(counts_df_true)*0.10))].to_list()
        
        return highexpgenes
    
    def feature_radial_decoding(self, use_svm = True):
        """
        This function will perform feature based radial decoding on all barcodes as reference. Dot sequences
        that appear n number of times defined by min seed will be kept. Three rounds of decoding can be 
        performed with expanding radii. Additional features include the decoding of highly expressed genes first
        and removing those dots for next iterative rounds of decoding. This algorithm also utilizes a SVM model
        to assign probabilities to spots on the liklihood that they are true dots. Random signals will most
        likely have a low probability score and can be removed prior to real decoding. Alternatively, setting
        probability cutoff to 0 will not perform any filtering and the outputed codeword score for each decoded
        spot can be used for filtering based on fpr cutoff set by the user.
        """
        #record start time
        start = time.time()
        #read in locations file and drop unnamed columns
        locations = pd.read_csv(self.location_path)
        unwanted_columns = []
        for names in locations.columns:
            if "Unnamed" in names:
                unwanted_columns.append(names)
        locations = locations.drop(unwanted_columns, axis=1)
        #remove extra hybs
        locations = locations[locations["hyb"] < self.hybs].reset_index(drop=True)
        #get total dot count for later
        total_dots = len(locations)
        #read in codebook
        codebook = pd.read_csv(self.codebook_path)
        codebook = codebook.set_index(codebook.columns[0])
        location_path_name = Path(self.location_path).name
        #collect z info
        z_info = location_path_name.split("_")[-1].replace(".csv","")
        
        #check to see if the necessary amount of hybs are present
        assert len(locations["hyb"].unique()) == self.hybs, "Locations file is missing a hyb"
        
        #make directories
        Path(self.output_dir).mkdir(parents=True, exist_ok = True)
        output_path = Path(self.output_dir) / f"diff_{self.diff}_minseed_{self.min_seed}_z_{z_info}_finalgenes.csv"
        
        #rough decoding to identify true and false dots
        #the parameters here is somewhat loose in order to collect fakes
        if self.parity_round:
            rough_diff = 1
        else:
            rough_diff = 0
        decoded_rough, indicies_used_1 = decode.radial_decoding_parallel(self, locations, codebook,
                                                                  num_barcodes=self.num_barcodes, radius=1.2, diff=rough_diff,cycle=1,
                                                                  min_seed=self.num_barcodes-self.diff, hybs = self.hybs, 
                                                                  include_undecoded = False, 
                                                                  look_for_parity = False, parity_round=self.parity_round)
        #only get the indicies of decoded genes (excluding undefined) and separate true and fake
        decoded_rough_fakes = decoded_rough[decoded_rough["genes"].str.startswith("fake")]
        decoded_rough_trues = decoded_rough.drop(decoded_rough_fakes.index)
        indicies_used_rough_trues = decoded_rough_trues.index.tolist()
        indicies_used_rough_fakes = decoded_rough_fakes.index.tolist()
        #isolate all dots used indicies in dot indicies list
        indicies_trues = list(map(indicies_used_1.__getitem__, indicies_used_rough_trues))
        indicies_fakes = list(map(indicies_used_1.__getitem__, indicies_used_rough_fakes))
        flattened_indicies_used_trues = [element for sublist in indicies_trues for element in sublist]
        flattened_indicies_used_fakes = [element for sublist in indicies_fakes for element in sublist]
        locations_trues = locations.iloc[flattened_indicies_used_trues].reset_index(drop=True)
        locations_fakes = locations.iloc[flattened_indicies_used_fakes].reset_index(drop=True)
        
        #assign probabilities
        if len(locations_fakes) < 500:
            print("Cannot generate probabilities due to low number of fake dots.")
        else:
            if use_svm == True:
                true_probs_only, X_test, y_test, locations, plt = decode.poly_rbf_gen_dot_probabilities(locations_trues,locations_fakes, locations)
                #median on probability for trues and fakes
                fake_median_prob = np.median(np.compress(y_test== -1,true_probs_only))
                true_median_prob = np.median(np.compress(y_test== 1,true_probs_only))
                #write out test set
                X_test.to_csv(str(output_path.parent/"test_set.csv"))
                y_test.to_csv(str(output_path.parent/"test_set_labels.csv"))
                np.savetxt(str(output_path.parent/"test_set_probabilities.csv"),true_probs_only)
                with open(str(output_path.parent/"median_probs.txt"), "w+") as f:
                    f.write(f"Median probability of On for observed fakes were {round(fake_median_prob,2)}.\n")
                    f.write(f"Median probability of On for observed trues were {round(true_median_prob,2)}.\n")
                    f.close()
                plt.savefig(str(output_path.parent/"feature_ranking.png"), dpi=300, bbox_inches = "tight")
                plt.clf()
                #probability cutoff, keep unused dots for later output
                cutoff_unused_dots = locations[locations["probability on"] <= self.probability_cutoff].reset_index(drop=True)
                locations = locations[locations["probability on"] > self.probability_cutoff].reset_index(drop=True)
                print("Will now begin decoding...")
        
        #only keep top 10% of highly expressed genes
        if self.decode_high_exp_genes == True:
            #run decoding first pass
            #we are enabling parity here, if applicable, to accurately obtain true codes
            decoded_1, indicies_used_1 = decode.radial_decoding_parallel(self, locations, codebook,
                                                                  num_barcodes=self.num_barcodes, radius=self.first_radius, diff=0,cycle=1,
                                                                  min_seed=self.high_exp_seed, hybs = self.hybs, include_undecoded = False,
                                                                  look_for_parity = True, parity_round=self.parity_round)
            #get highly expressed genes
            highexpgenes = decode.return_highly_expressed_names(decoded_1)
            #initialize loop
            highexpgenes_2 = highexpgenes.copy()
            locations_temp = locations.copy()
            #record of highly expressed decoded genes
            record_genes = []
            record_true_dots_used = []
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
                record_true_dots_used.append(locations_temp.iloc[flattened_indicies_used])
                #remove already decoded dots
                locations_temp = locations_temp.drop(flattened_indicies_used).reset_index(drop=True)
                #run decoding with defined pixel search
                decoded_1, indicies_used_1 = decode.radial_decoding_parallel(self, locations_temp, codebook,
                                                                      num_barcodes=self.num_barcodes, radius=self.first_radius, diff=0,cycle=1,
                                                                      min_seed=self.high_exp_seed, hybs = self.hybs, 
                                                                      include_undecoded = False, look_for_parity = True,
                                                                      parity_round=self.parity_round)
                #get new highly expressed gene list
                highexpgenes_2 = decode.return_highly_expressed_names(decoded_1)
            #combine final highly expressed genes
            decoded_1 = pd.concat(record_genes).reset_index(drop=True)
            #use locations temp after exitting loop for second round
            new_locations = locations_temp.copy()
            del locations_temp
            
        else:
            #run decoding first pass
            #giving preference to parity codes since those will be most accurate
            decoded_1, indicies_used_1 = decode.radial_decoding_parallel(self, locations, codebook,
                                                                  num_barcodes=self.num_barcodes, radius=self.first_radius, diff=0,cycle=1,
                                                                  min_seed=self.min_seed, hybs = self.hybs, include_undecoded = False,
                                                                  look_for_parity = True, parity_round=self.parity_round)
            #only get the indicies of decoded genes (excluding undefined) and separate true and fake
            decoded_1_fakes = decoded_1[decoded_1["genes"].str.startswith("fake")]
            decoded_1_trues = decoded_1.drop(decoded_1_fakes.index)
            indicies_used_df_trues = decoded_1_trues.index.tolist()
            indicies_used_df_fakes = decoded_1_fakes.index.tolist()
            indicies_used_df = decoded_1.index.tolist()
            #isolate all dots used indicies in dot indicies list
            indicies_used_1_trues = list(map(indicies_used_1.__getitem__, indicies_used_df_trues))
            indicies_used_1_fakes = list(map(indicies_used_1.__getitem__, indicies_used_df_fakes))
            indicies_used_1 = list(map(indicies_used_1.__getitem__, indicies_used_df))
            #flatten 1st set of indicies used list
            flattened_indicies_used = [element for sublist in indicies_used_1 for element in sublist]
            #remove already decoded dots
            new_locations = locations.drop(flattened_indicies_used).reset_index(drop=True)
            
        #output results from first pass
        decoded_1.sort_values("genes").to_csv(str(output_path).replace("finalgenes","round1"))
        
        #run decoding second pass with same or different search radius
        if self.triple_decode == True:
            decoded_2, indicies_used_2 = decode.radial_decoding_parallel(self, new_locations, codebook,
                                                              num_barcodes=self.num_barcodes, radius=self.second_radius, diff=0, cycle=2,
                                                              min_seed=self.min_seed, hybs = self.hybs, include_undecoded = True, 
                                                              look_for_parity = False,
                                                              parity_round=self.parity_round)
        else:
            decoded_2, indicies_used_2 = decode.radial_decoding_parallel(self, new_locations, codebook,
                                                              num_barcodes=self.num_barcodes, radius=self.second_radius, diff=self.diff, cycle=2,
                                                              min_seed=self.min_seed, hybs = self.hybs, include_undecoded = True, 
                                                              look_for_parity = False,
                                                              parity_round=self.parity_round)
            
        if self.triple_decode == True:
            #output results from second pass
            decoded_combined = pd.concat([decoded_1, decoded_2])
            decoded_combined = decoded_combined[decoded_combined["genes"] != "Undefined"]
            decoded_combined.sort_values("genes").reset_index(drop=True).to_csv(str(output_path).replace("finalgenes","round2"))
            #isolate undefined 
            decoded_2_undefined = decoded_2[decoded_2["genes"] == "Undefined"]
            #remove undefined 
            decoded_2 = decoded_2[decoded_2["genes"] != "Undefined"]
            #get fakes
            decoded_2_fakes = decoded_2[decoded_2["genes"].str.startswith("fake")]
            #only trues
            decoded_2_trues = decoded_2.drop(decoded_2_fakes.index)
            #get indicies for trues, fakes, and undefined
            indicies_used_df_trues = decoded_2_trues.index.tolist()
            indicies_used_df_fakes = decoded_2_fakes.index.tolist()
            indicies_used_df_undefined = decoded_2_undefined.index.tolist()
            indicies_used_df = decoded_2.index.tolist()
            #isolate used indicies in dot indicies list
            indicies_used_2_trues = list(map(indicies_used_2.__getitem__, indicies_used_df_trues))
            indicies_used_2_fakes = list(map(indicies_used_2.__getitem__, indicies_used_df_fakes))
            indicies_used_2_undefined = list(map(indicies_used_2.__getitem__, indicies_used_df_undefined))
            indicies_used_2 = list(map(indicies_used_2.__getitem__, indicies_used_df))
            #flatten 2nd set of indicies used list
            flattened_indicies_used_2 = [element for sublist in indicies_used_2 for element in sublist]
            #remove already decoded dots
            new_locations_2 = new_locations.drop(flattened_indicies_used_2).reset_index(drop=True)
    
            #run decoding third pass
            try:
                decoded_3, indicies_used_3 = decode.radial_decoding_parallel(self, new_locations_2, codebook,
                            num_barcodes=self.num_barcodes, radius=self.third_radius,diff=self.diff,cycle=3,
                            min_seed=self.min_seed, hybs = self.hybs, include_undecoded = True, look_for_parity = False,
                            parity_round=self.parity_round)
                #combine decoded dfs
                decoded_combined = pd.concat([decoded_1, decoded_2, decoded_3])
                if self.include_undefined == False:
                    decoded_combined = decoded_combined[decoded_combined["genes"] != "Undefined"]
            except:
                with open("decoding_message.txt", "w+") as f:
                    f.write("Triple decoding did not yield anything.")
                    f.close()
                decoded_combined = pd.concat([decoded_1, decoded_2])  
                if self.include_undefined == False:
                    decoded_combined = decoded_combined[decoded_combined["genes"] != "Undefined"]       
        else:
            #combine decoded dfs
            decoded_combined = pd.concat([decoded_1, decoded_2])
            if self.include_undefined == False:
                decoded_combined = decoded_combined[decoded_combined["genes"] != "Undefined"]    
            #separate true, fake and undefined
            decoded_2_undefined = decoded_2[decoded_2["genes"] == "Undefined"]
            decoded_2_fakes = decoded_2[decoded_2["genes"].str.startswith("fake")]
            decoded_2_trues = decoded_2.drop(decoded_2_fakes.index).drop(decoded_2_undefined.index)
            indicies_used_df_trues = decoded_2_trues.index.tolist()
            indicies_used_df_fakes = decoded_2_fakes.index.tolist()
            indicies_used_df_undefined = decoded_2_undefined.index.tolist()
            #isolate used indicies in dot indicies list
            indicies_used_2_trues = list(map(indicies_used_2.__getitem__, indicies_used_df_trues))
            indicies_used_2_fakes = list(map(indicies_used_2.__getitem__, indicies_used_df_fakes))
            indicies_used_2_undefined = list(map(indicies_used_2.__getitem__, indicies_used_df_undefined))
    
        #sort and reset index
        final_decoded = decoded_combined.sort_values("genes").reset_index(drop=True)
        final_decoded.to_csv(str(output_path.parent / f"diff_{self.diff}_minseed_{self.min_seed}_z_{z_info}_unfiltered.csv"))
        
        #filter by desired fdr
        final_decoded, cutoff_plt = decode.set_fpr(final_decoded, codebook, fpr_cutoff=self.desired_fdr)
        
        #write files
        final_decoded.to_csv(str(output_path))
        cutoff_plt.savefig(str(output_path.parent/"fdr.png"), dpi=300, bbox_inches = "tight")
        
        #output used and unused dot locations
        #---------------------------------------------------------------------------------------------
        #collection dots used in round 1 decoding
        if self.decode_high_exp_genes == True:
            locations1_trues = pd.concat(record_true_dots_used).reset_index(drop=True)
            locations1_fakes = pd.DataFrame()
        else:
            flattened_indicies_used_trues_1 = [element for sublist in indicies_used_1_trues for element in sublist]
            flattened_indicies_used_fakes_1 = [element for sublist in indicies_used_1_fakes for element in sublist]
            locations1_trues = locations.iloc[flattened_indicies_used_trues_1]
            locations1_fakes = locations.iloc[flattened_indicies_used_fakes_1]
            
        #collect dots used in second round
        if self.triple_decode == False:
            #flatten 2nd set of indicies used list
            flattened_indicies_used_2 = [element for sublist in indicies_used_2 for element in sublist]
           
        #flatten second set of indicies used list
        flattened_indicies_used_trues_2 = [element for sublist in indicies_used_2_trues for element in sublist]
        flattened_indicies_used_fakes_2 = [element for sublist in indicies_used_2_fakes for element in sublist]
        flattened_indicies_used_undefined_2 = [element for sublist in indicies_used_2_undefined for element in sublist]
        #collect dots
        locations2_trues = new_locations.iloc[flattened_indicies_used_trues_2]
        locations2_fakes = new_locations.iloc[flattened_indicies_used_fakes_2]
        locations2_undefined = new_locations.iloc[flattened_indicies_used_undefined_2]
        
        if self.triple_decode == True:
            #in case there was no output from triple decode
            try:
                #separate fakes, and trues
                decoded_3_undefined = decoded_3[decoded_3["genes"] == "Undefined"]
                decoded_3_fakes = decoded_3[decoded_3["genes"].str.startswith("fake")]
                decoded_3_trues = decoded_3.drop(decoded_3_fakes.index).drop(decoded_3_undefined.index)
                indicies_used_df_trues = decoded_3_trues.index.tolist()
                indicies_used_df_fakes = decoded_3_fakes.index.tolist()
                indicies_used_df_undefined = decoded_3_undefined.index.tolist()
                #isolate used indicies in dot indicies list
                indicies_used_3_trues = list(map(indicies_used_3.__getitem__, indicies_used_df_trues))
                indicies_used_3_fakes = list(map(indicies_used_3.__getitem__, indicies_used_df_fakes))
                indicies_used_3_undefined = list(map(indicies_used_3.__getitem__, indicies_used_df_undefined))
                #flatten 3rd set of indicies used list
                flattened_indicies_used_trues_3 = [element for sublist in indicies_used_3_trues for element in sublist]
                flattened_indicies_used_fakes_3 = [element for sublist in indicies_used_3_fakes for element in sublist]
                flattened_indicies_used_undefined_3 = [element for sublist in indicies_used_3_undefined for element in sublist]
                flattened_indicies_used_3 = [element for sublist in indicies_used_3 for element in sublist]
                #collect dots
                locations3_trues = new_locations_2.iloc[flattened_indicies_used_trues_3]
                locations3_fakes = new_locations_2.iloc[flattened_indicies_used_fakes_3]
                dots_used_undefined = new_locations_2.iloc[flattened_indicies_used_undefined_3].reset_index(drop=True)
                #combine used dots
                dots_used_trues = pd.concat([locations1_trues,locations2_trues,locations3_trues]).reset_index(drop=True)
                dots_used_fakes = pd.concat([locations1_fakes,locations2_fakes,locations3_fakes]).reset_index(drop=True)
                #remove used dots (which includes trues,fakes, and undefineds)
                dots_unused = new_locations_2.drop(flattened_indicies_used_3).reset_index(drop=True)
                if "cutoff_unused_dots" in locals():
                    dots_unused = pd.concat([dots_unused,cutoff_unused_dots]).reset_index(drop=True)
                #write out used dots
                dots_used_trues.to_csv(str(output_path.parent / f"dots_used_trues_z_{z_info}.csv"))
                dots_used_fakes.to_csv(str(output_path.parent / f"dots_used_fakes_z_{z_info}.csv"))
                dots_used_undefined.to_csv(str(output_path.parent / f"dots_used_undefined_z_{z_info}.csv"))
                #write out unused dots
                dots_unused.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
            except:
                dots_used_trues = pd.concat([locations1_trues,locations2_trues]).reset_index(drop=True)
                dots_used_fakes = pd.concat([locations1_fakes,locations2_fakes]).reset_index(drop=True)
                #write out used dots (which includes trues,fakes, and undefineds)
                dots_used_trues.to_csv(str(output_path.parent / f"dots_used_trues_z_{z_info}.csv"))
                dots_used_fakes.to_csv(str(output_path.parent / f"dots_used_fakes_z_{z_info}.csv")) 
                locations2_undefined.to_csv(str(output_path.parent / f"dots_used_undefined_z_{z_info}.csv"))
                #write out unused dots
                if "cutoff_unused_dots" in locals():
                    dots_unused = pd.concat([new_locations_2,cutoff_unused_dots]).reset_index(drop=True)
                new_locations_2.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
        else:
            #remove already decoded dots
            dots_unused = new_locations.drop(flattened_indicies_used_2).reset_index(drop=True)
            #combined dots used
            dots_used_trues = pd.concat([locations1_trues,locations2_trues]).reset_index(drop=True)
            dots_used_fakes = pd.concat([locations1_fakes,locations2_fakes]).reset_index(drop=True)
            #write out used dots (which includes trues,fakes, and undefineds)
            dots_used_trues.to_csv(str(output_path.parent / f"dots_used_trues_z_{z_info}.csv"))
            dots_used_fakes.to_csv(str(output_path.parent / f"dots_used_fakes_z_{z_info}.csv"))
            locations2_undefined.to_csv(str(output_path.parent / f"dots_used_undefined_z_{z_info}.csv"))
            #write out unused dots
            if "cutoff_unused_dots" in locals():
                dots_unused = pd.concat([dots_unused,cutoff_unused_dots]).reset_index(drop=True)
            dots_unused.to_csv(str(output_path.parent / f"dots_unused_z_{z_info}.csv"))
            
        #Output percent decoded
        
        percent_decoded = str(round((((len(dots_used_trues)+len(dots_used_fakes))/total_dots) * 100),2))
        percent_output = output_path.parent / f"percent_decoded_z_{z_info}.txt"
        with open(str(percent_output),"w+") as f:
            f.write(f'Percent of dots decoded = {percent_decoded}')
            f.close()
        
        print(f"This task took {round((time.time()-start)/60,2)} min")
                         
