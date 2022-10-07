import numpy as np
from tqdm import tqdm

def nuclear_cyto_stitching(cyto, nuc, threshold=0.10):
    """
    Stitch cyto masks and nuclear masks to have same cell ids.
    Parameters
    ----------
    cyto=list of arrays or single cyto array
    nuc=list of arrays or single nuc array
    threshold=percent overlap
    """
    
    if type(cyto) != list:
        
        #make copy of mask to not overwrite original
        cyto_new = np.copy(cyto)
        nuc_new = np.copy(nuc)
        
        #stitch masks (brute force)
        for i in range(cyto_new.shape[0]):
            for j in range(cyto_new.shape[1]):
                if cyto_new[j,i] == 0:
                    continue
                else:
                    if nuc_new[j,i] > 0:
                        nuc_new[j,i] = cyto_new[j,i]
                    else:
                        continue
                
        #converst masks to only one of the cells
        for i in np.compress(np.unique(cyto_new)>0,np.unique(cyto_new)):
            arr1_int = (cyto_new==i).astype(int)
            arr2_int = (nuc_new>0).astype(int)

            #compare masks
            matched_counts = np.where((arr1_int==1)& (arr2_int==1))
            cyto_area = np.count_nonzero(arr1_int == 1)
            percent = len(matched_counts[0])/cyto_area

            #if percent overlap is greater than threshold keep, else throw away
            if percent < threshold:
                cyto_new[cyto_new==i]=0
                nuc_new[nuc_new==i]=0
                  
        return cyto_new, nuc_new
    
    else:
        #for new masks
        new_cyto_arr = []
        new_nuc_arr = []
        for i in tqdm(range(len(cyto))):
            #make copy of mask to not overwrite original
            cyto_new = np.copy(cyto[i])
            nuc_new = np.copy(nuc[i])
            #stitch masks (brute force)
            for k in range(cyto_new.shape[0]):
                for j in range(cyto_new.shape[1]):
                    if cyto_new[j,k] == 0:
                        continue
                    else:
                        if nuc_new[j,k] > 0:
                            nuc_new[j,k] = cyto_new[j,k]
                        else:
                            continue
           #converst masks to only one of the cells
            for _ in np.compress(np.unique(cyto_new)>0,np.unique(cyto_new)):
                arr1_int = (cyto_new==_).astype(int)
                arr2_int = (nuc_new>0).astype(int)

                #compare masks
                matched_counts = np.where((arr1_int==1)& (arr2_int==1))
                cyto_area = np.count_nonzero(arr1_int == 1)
                percent = len(matched_counts[0])/cyto_area

                #if percent overlap is greater than threshold keep, else throw away
                if percent < threshold:
                    cyto_new[cyto_new==_]=0
                    nuc_new[nuc_new==_]=0
           
            new_cyto_arr.append(cyto_new)
            new_nuc_arr.append(nuc_new)
            
        return new_cyto_arr, new_nuc_arr