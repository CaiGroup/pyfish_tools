"""
author: Katsuya Lex Colon
updated: 12/03/21
group: Cai Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def percent_false_positive(df, codebook, fakebook):
    """calculate percent false positive
    Parameters
    ----------
    df = dataframe of gene by cell
    codebook = codebook of only real genes
    fakebook = codebook of fake genes
    
    """
    #get cell ids
    cells = df.columns
    #make fake barcodes df
    fakebrcds = df[df.index.str.startswith("fake")]
    #make real barcodes df
    real = df.drop(fakebrcds.index, axis=0)
    #calculate percent false positive in each cell
    fp_list = []
    M_on = len(codebook)
    M_off = len(fakebook)
    for i in cells:
        #get percent fakes per cell
        N_off = fakebrcds[i].sum()
        N_on = real[i].sum()
        percent_fp_raw = (N_off/(N_off+N_on))
        #false positive rate
        false_count_freq = N_off/M_off
        false_positive_counts = M_on*false_count_freq
        norm_false_positive_rate = false_positive_counts/N_on
        fp_list.append([i, N_off+N_on,N_off,N_on,percent_fp_raw, norm_false_positive_rate])
        
    #average barcodes per cell
    fake_avg = fakebrcds.mean(axis=1)
    real_avg = real.mean(axis=1)
    comb_avg = pd.concat([fake_avg,real_avg])
    comb_sorted = comb_avg.sort_values(ascending=False)
        
    #make new df
    new_df = pd.DataFrame(fp_list)
    new_df.columns = ["cell name", "total_counts","total_fake","total_real", "FP raw", "FDR"]
    
    #make on and off target plot
    plt.fill_between(np.arange(0,len(new_df),1), new_df["total_counts"].sort_values(ascending=False), label = "On Target")
    plt.fill_between(np.arange(0,len(new_df),1), new_df["total_fake"].sort_values(ascending=False), color="red", label = "Off Target")
    plt.legend()
    plt.xlabel("Cells", fontsize=12)
    plt.ylabel("Total Counts", fontsize=12)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    sns.despine()
    plt.show()
    
    #make average barcode counts per cell plot
    color= ["red" if i == True else "blue" for i in comb_sorted.index.str.startswith("fake")]
    plt.scatter(np.arange(0,len(comb_sorted.values),1), comb_sorted.values, color= color, s=3)
    plt.xlabel("Barcodes", fontsize=12)
    plt.ylabel("Average Counts per Cell", fontsize=12)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    sns.despine()
    plt.show()

    return new_df, fake_avg, norm_false_positive_rate