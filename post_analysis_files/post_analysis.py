"""
author: Katsuya Lex Colon
updated: 06/17/22
group: Cai Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.pyplot import figure

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

    return new_df, fake_avg

def correlation(mtx,mtx_ref, label_x=None, label_y=None, title=None, 
                cell_size_normalized=True,return_comb_df=False,
                log=False):
    
    """
    Output correlation plot
    
    Parameters
    ---------
    mtx: gene by cell matrix
    mtx_ref: gene by cell matrix of the data we are comparing against
    label_x: string for x label
    label_y: string for y label
    title: string for title
    cell_size_normalized: bool on whether the data was cell size normalized
    return_comb_df: bool to return merged dataframe for correlation
    """

    #convert data to pseudobulk rnaseq data
    bulk = pd.DataFrame(mtx.mean(axis=1)).reset_index()
    bulk.columns = ["Genes", "Counts"]
    bulk["Genes"] = bulk["Genes"].str.lower()

    bulk_ref = pd.DataFrame(mtx_ref.mean(axis=1)).reset_index()
    bulk_ref.columns = ["Genes", "Counts ref"]
    bulk_ref["Genes"] = bulk_ref["Genes"].str.lower()

    #merge dfs
    comb_2 = pd.merge(bulk_ref,bulk)
    comb_2 = comb_2.drop(comb_2[comb_2["Genes"].str.startswith("fake")].index)
    
    #perform linear regression
    x = comb_2["Counts ref"].values
    x_t = np.vstack([x, np.zeros(len(x))]).T
    y = comb_2["Counts"].values
    m,c = np.linalg.lstsq(x_t, y, rcond=None)[0]
    
    #get pearsons r
    r = pearsonr(x,y)[0]
    
    if log == False:
        #make plot
        figure(figsize=(4,5), dpi=80)
        #show smfish correlation
        plt.plot(x, y, 'bo', alpha=0.5)
        ytick_interval = plt.yticks()[0][1]-plt.yticks()[0][0]
        plt.plot(x, x*m, c = "k")
        plt.title(title, fontweight="bold")
        if cell_size_normalized==True:
            plt.xlabel(f"{label_x} average counts/$\mu m^{2}$")
            plt.ylabel(f"{label_y} average counts/$\mu m^{2}$")
        else:
            plt.xlabel(f"{label_x} average counts")
            plt.ylabel(f"{label_y} average counts")
        max_y = max(max(x*m),max(y))
        anno_pear = (min(x),max_y-(ytick_interval/2))
        anno_r = (min(x),max_y-ytick_interval)
        plt.annotate(f"Pearson's r= {round(r,2)}", anno_pear, fontsize=12)
        plt.annotate(f"Efficiency = {round(m,2)}", anno_r, fontsize=12)
        sns.despine()
        plt.show()
    
    else:
        figure(figsize=(4,5), dpi=80)
        #plot correlation and efficiency
        plt.scatter(x = np.log2(comb_2["Counts ref"]), y = np.log2(comb_2["Counts"]), c="b", alpha=0.5)
        ytick_interval = plt.yticks()[0][1]-plt.yticks()[0][0]
        max_y = max(np.log2(comb_2["Counts"]).values)
        plt.xlabel(f"Log2({label_x} average counts)", fontsize=12)
        plt.ylabel(f"Log2({label_y} counts)", fontsize=12)
        anno_pear = (min(np.log2(comb_2["Counts ref"]).values),max_y-(ytick_interval/2))
        anno_r = (min(np.log2(comb_2["Counts ref"]).values),max_y-ytick_interval)
        plt.annotate(f"Pearson's r= {round(r,2)}", anno_pear, fontsize=12)
        plt.annotate(f"Efficiency = {round(m,2)}", anno_r, fontsize=12)
        plt.title(title, fontweight="bold")
        sns.despine()
        plt.show()
    
    if return_comb_df == True:
        return comb_2