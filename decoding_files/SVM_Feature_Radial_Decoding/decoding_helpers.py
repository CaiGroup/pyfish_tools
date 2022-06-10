"""
author: Katsuya Lex Colon
group: Cai Lab
date: 05/12/22
"""
#general packages
import pandas as pd
import numpy as np
# SVM model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#for splitting training and test set
from sklearn.model_selection import train_test_split
#screening hyperparameters
from sklearn.model_selection import GridSearchCV
#scaling features
from sklearn.preprocessing import StandardScaler
#calibrate model for probability predictions (required for LinearSVC)
from sklearn.calibration import CalibratedClassifierCV
#metrics module for model accuracy calculation
from sklearn import metrics
#output most important features
from sklearn.inspection import permutation_importance
#plotting package
import matplotlib.pyplot as plt
import seaborn as sns
#random modules
import time
import warnings
warnings.filterwarnings("ignore")

def lin_gen_dot_probabilities(dots_used_trues,dots_used_fakes, locations):
    """
    A linear SVM classifier to assign probability scores to dots. 
    This model should be faster for larger datasets, but less accurate.
    
    Parameters
    ----------
    dots_used_trues: dot locations of true dots with features to train svm classifier
    dots_used_fakes: dot locations of fake dots with features to train svm classifier
    locations: all of the dots
    """

    #get attributes
    trues = dots_used_trues[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
    trues["labels"] = 1
    fakes = dots_used_fakes[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
    fakes["labels"] = -1
    
    #make sure number of fakes is > 100
    assert len(fakes) >= 100, "Not enough fake dots to train."
    
    #downsample trues to match fakes
    #generally there will be less fakes then trues 
    trues = trues.iloc[:len(fakes),:]

    #combine trues and fakes into one array
    comb = pd.concat([trues,fakes]).reset_index(drop=True)

    #Split dataset into training set and test set
    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(comb.iloc[:,0:6], 
                                                        comb["labels"], test_size=0.3,random_state=42, 
                                                        stratify=comb["labels"]) 

    ### Train classifiers
    start = time.time()

    #z score normalize
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    print(f"Number of features in dataset = {scaler.n_features_in_}")
    print(f"Mean of features = {scaler.mean_}")
    print(f"Variance of features = {scaler.var_}")

    #parameters to test
    C_range = np.logspace(-2, 10, 20)
    param_grid = dict(C=C_range.tolist(), 
                      loss = ["hinge","squared_hinge"],
                      penalty = ["l1", "l2"])
    
    #total number of cross validations
    total_cross_val = 5
    
    #total number of fits
    total_fits = (len(C_range) * len(["hinge","squared_hinge"]) * len(["l1", "l2"])) * total_cross_val
    
    print(f"Performing hyperparameter optimization totalling {total_fits} fits...")
    
    #screen all permutation of parameters
    grid = GridSearchCV(LinearSVC(max_iter=1e5, tol=1e-4, random_state=42),
                        param_grid=param_grid, cv=total_cross_val,  n_jobs=-1)
    grid.fit(X_scaled, y_train)

    #which is the best
    print("The best parameters are %s with a score of %0.2f on average" % (grid.best_params_, grid.best_score_))
    print(f"Model building took {round((time.time()-start)/60, 2)} min and is now being applied to test set.")
    #Calibrate estimator using the best parameters
    clf = CalibratedClassifierCV(grid.best_estimator_, cv="prefit")
    #refit classifier
    clf = clf.fit(X_scaled, y_train)
    # Model Accuracy: how often is the classifier correct with its labels
    #scale test data using training mean and std
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    print(f"Label classification accuracy: {round(metrics.accuracy_score(y_test, y_pred), 2)}")
    print("Generating probabilities for each dot...")
    #obtain probabilities
    y_proba = clf.predict_proba(X_test_scaled)
    true_probs_only = y_proba[:,1]

    #get probabilities for every dot
    locations_features = locations[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
    locations_scaled = scaler.transform(locations_features)
    loc_proba = clf.predict_proba(locations_scaled)
    locations["probability on"] = loc_proba[:,1]
    
    print(f"Probabilities assigned, Total time = {round((time.time()-start)/60, 2)} min")
    
    return true_probs_only, X_test, y_test, locations


def rbf_gen_dot_probabilities(dots_used_trues,dots_used_fakes, locations):
    """
    A SVM classifier using a radial basis function kernel to assign probability scores to dots.
    This model tends to be slower for larger datasets, but more accurate.
    
    Parameters
    ----------
    dots_used_trues: dot locations of true dots with features to train svm classifier
    dots_used_fakes: dot locations of fake dots with features to train svm classifier
    locations: all of the dots
    """
    #get attributes
    trues = dots_used_trues[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
    trues["labels"] = 1
    fakes = dots_used_fakes[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
    fakes["labels"] = -1
    
    #make sure number of fakes is > 100
    assert len(fakes) >= 100, "Not enough fake dots to train."
    
    #if there is over 500,000 fake spots then down sample for training
    fakes = fakes.iloc[:500000,:]
    
    #downsample trues to match fakes
    #generally there will be less fakes then trues 
    trues = trues.iloc[:len(fakes),:]

    #combine trues and fakes into one array
    comb = pd.concat([trues,fakes]).reset_index(drop=True)

    #Split dataset into training set and test set
    # 70% training and 30% test
    X_train, X_test, y_train, y_test = train_test_split(comb.iloc[:,0:6], 
                                                        comb["labels"], test_size=0.30,random_state=42, 
                                                        stratify=comb["labels"])

    # Train classifiers
    start = time.time()

    #z score normalize
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    print(f"Number of features in dataset = {scaler.n_features_in_}")
    print(f"Mean of features = {scaler.mean_}")
    print(f"Variance of features = {scaler.var_}")

    #parameters to test
    C_range = np.logspace(-2, 10, 12)
    gamma_range = np.logspace(-9, 3, 12)
    param_grid = dict(gamma=gamma_range, C=C_range)
                                  
    #total number of cross validations
    total_cross_val = 5
    
    #total number of fits
    total_fits = len(C_range) * len(gamma_range) * total_cross_val
                                  
    print(f"Performing hyperparameter optimization totalling {total_fits} fits...")
                                  
    #screen all permutation of parameters
    grid = GridSearchCV(SVC(kernel="rbf", max_iter=2e5, tol=1e-3, random_state=42), 
                        param_grid=param_grid, cv=total_cross_val,  n_jobs=-1,)
    grid.fit(X_scaled, y_train)

    #which is the best
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    print(f"Model building took {round((time.time()-start)/60, 2)} min and is now being applied to test set.")
    #generate probability model based on best params
    clf = SVC(kernel="rbf", max_iter=1e6, tol=1e-3, 
              random_state=42,gamma = grid.best_params_["gamma"], 
              C=grid.best_params_["C"], probability=True)
    clf.fit(X_scaled, y_train)
    # Model Accuracy: how often is the classifier correct with its labels
    #scale test data using training mean and std
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    print(f"Label classification accuracy: {round(metrics.accuracy_score(y_test, y_pred), 2)}")
    print("Generating probabilities for each dot...")
    #obtain probabilities
    y_proba = clf.predict_proba(X_test_scaled)
    true_probs_only = y_proba[:,1]
    
    #get probabilities for every dot
    locations_features = locations[["flux","max intensity","size", "sharpness", "symmetry","roundness by gaussian fits"]]
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
    gene_locations = decoded gene locations
    truebook = codebook of only real genes
    fakebook = codebook of fake genes
    
    """
    #make fake barcodes df
    fakebrcds = gene_locations[gene_locations["genes"].str.startswith("fake")]
    
    #make real barcodes df
    real = gene_locations.drop(fakebrcds.index, axis=0)
    
    fdr_list = []
    #calculate fdr per cell if availible
    if "cell number" in gene_locations.columns:
        #false positive rate
        M_on = len(truebook)
        M_off = len(fakebook)
        cell_ids = gene_locations["cell number"].unique()
        for cell in cell_ids:
            #get percent fakes per cell
            N_off = len(fakebrcds[fakebrcds["cell number"]==cell])
            N_on = len(real[real["cell number"]==cell])
            if N_on >= 1 :
                false_count_freq = N_off/M_off
                false_positive_counts = M_on*false_count_freq
                norm_false_positive_rate = false_positive_counts/N_on
                fdr_list.append(norm_false_positive_rate)
            else:
                continue
        fdr = np.mean(fdr_list)
    else:
        #false positive rate
        M_on = len(truebook)
        M_off = len(fakebook)
        N_on = len(real)
        N_off = len(fakebrcds)
        false_count_freq = N_off/M_off
        false_positive_counts = M_on*false_count_freq
        fdr = false_positive_counts/N_on   
    
    return fdr

def set_fdr(gene_locations, codebook, fdr_cutoff=0.05):
    """
    Function to sort barcodes by codeword score and sample subsets of decreasing codeword score
    while calculating fdr. A final gene locations file will be outputted based on user defined
    fdr_cutoff.
    
    Parameters
    ----------
    gene_locations: decoded gene locations file
    fdr_cutoff: desired fdr value
    """
    #sort barcodes by codeword score
    sorted_genes = gene_locations.sort_values("codeword score", ascending=False).reset_index(drop=True)
    
    #separate true and fake codebook
    fakebook = codebook[codebook.index.str.startswith("fake")]
    truebook = codebook.drop(fakebook.index)
    
    #calculate fdr while sampling barcodes
    collect_fdr = []
    for i in np.linspace(100, len(sorted_genes),1000).astype(int):
        iso_loc = sorted_genes.iloc[0:i]
        #do not include fakes generated from two separate masks
        if "cell number" in iso_loc.columns:
            iso_loc = iso_loc[iso_loc["cell number"].astype(int) == iso_loc["cell number"]].reset_index(drop=True)
        #calculate fdr
        fdr_val = false_positive_rate(iso_loc, truebook, fakebook)
        collect_fdr.append([i,fdr_val])  
     
    #convert to array
    collect_fdr = np.array(collect_fdr)
    
    #get fdr below cutoff with the highest number of decoded barcodes
    filtered_by_fdr = collect_fdr[np.where(collect_fdr[:,1]<=fdr_cutoff)[0]]
    #in case nothing is below cutoff then return empty df
    if filtered_by_fdr.size == 0:
        desired_df = pd.DataFrame()
        #generate plot
        plt.plot(collect_fdr[:,0],collect_fdr[:,1])
        plt.xlabel("Decoded Barcodes")
        plt.ylabel("FDR")
        if fdr_cutoff != None:
            plt.axhline(fdr_cutoff, c="red", ls = "--", label="FDR cutoff")
        plt.legend()
        sns.despine()
    else:
        max_dots = filtered_by_fdr[np.argmax(filtered_by_fdr[:,0])]
        desired_df = sorted_genes.iloc[0:int(max_dots[0])]
        #generate plot
        plt.plot(collect_fdr[:,0],collect_fdr[:,1])
        plt.xlabel("Decoded Barcodes")
        plt.ylabel("FDR")
        if fdr_cutoff != None:
            plt.axhline(fdr_cutoff, c="red", ls = "--", label="FDR cutoff")
        plt.scatter(max_dots[0],max_dots[1], color = "k", label="Best")
        plt.legend()
        sns.despine()
    
    return desired_df,plt
