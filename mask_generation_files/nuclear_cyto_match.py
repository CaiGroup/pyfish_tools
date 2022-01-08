"""
author: Katsuya Lex Colon
updated: 12/03/21
group: Cai Lab
"""

import numpy as np
from tqdm import tqdm

def nuclear_cyto_matching(cyto, nuc, threshold=0.20):
    """Match cyto masks and nuclear masks. Keep cyto masks that have nucleus
    Parameters
    ----------
    cyto=list of arrays or single cyto array
    nuc=list of arrays or single nuc array
    threshold=percent overlap"""
    
    if type(cyto) != list:
        #make copy of mask to not overwrite original
        cyto_new = np.copy(cyto)
        #converst masks to only one of the cells
        for i in np.arange(1, len(np.unique(cyto)),1):
            arr1_int = (cyto==i).astype(int)
            arr2_int = (nuc>0).astype(int)

            #compare masks
            matched_counts = np.where((arr1_int==1)& (arr2_int==1))
            total_count = np.count_nonzero(arr1_int == 1)
            percent = len(matched_counts[0])/total_count

            #if percent overlap is greater than threshold keep, else throw away
            if percent < threshold:
                cyto_new[cyto_new==i]=0
                
        #get array of old number assignment
        new_numbers = np.arange(0,len(np.unique(cyto_new)),1)
            
        #changes old number assignments to new
        for i in range(len(np.unique(cyto_new))):
            if i !=0:
                old_number = np.unique(cyto_new)
                cyto_new[cyto_new==old_number[i]]=new_numbers[i]
                        
        return cyto_new
    
    else:
        #for new masks
        new_arr = []
        for i in tqdm(range(len(cyto))):
            #make copy of mask to not overwrite original
            cyto_new = np.copy(cyto[i])
            #converst masks to only one of the cells
            for j in np.arange(1, len(np.unique(cyto[i])),1):
                arr1_int = (cyto[i]==j).astype(int)
                arr2_int = (nuc[i]>0).astype(int)

                #compare masks
                matched_counts = np.where((arr1_int==1)& (arr2_int==1))
                total_count = np.count_nonzero(arr1_int == 1)
                percent = len(matched_counts[0])/total_count

                #if percent overlap is greater than threshold keep else throw away
                if percent < threshold:
                    cyto_new[cyto_new==j]=0
            #get array of old number assignment
            new_numbers = np.arange(0,len(np.unique(cyto_new)),1)
            
            #changes old number assignments to new
            for k in range(len(np.unique(cyto_new))):
                if k !=0:
                    old_number = np.unique(cyto_new)
                    cyto_new[cyto_new==old_number[k]]=new_numbers[k]
            new_arr.append(cyto_new)
        return new_arr