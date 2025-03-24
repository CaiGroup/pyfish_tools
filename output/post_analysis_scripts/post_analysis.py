"""
author: Katsuya Lex Colon
updated: 06/17/22
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.pyplot import figure
from scipy.stats import gaussian_kde
from matplotlib.font_manager import FontProperties

def percent_false_positive(df, codebook, fakebook):
    """calculate percent false positive
    Parameters
    ----------
    df = dataframe of gene by cell
    codebook = codebook of only real genes
    fakebook = codebook of fake genes
    
    """
    # Get cell ids
    cells = df.columns
    # Separate fake and real barcodes
    fake_barcodes = df[df.index.str.startswith("fake")]
    real_barcodes = df.drop(fake_barcodes.index, axis=0)
    # Calculate percent false positive in each cell
    fp_list = []
    M_on = len(codebook)
    M_off = len(fakebook)
    for i in cells:
        # Get percent fakes per cell
        N_off = fake_barcodes[i].sum()
        N_on = real_barcodes[i].sum()
        percent_fp_raw = (N_off / (N_off + N_on)) if (N_off + N_on) > 0 else 0
        # False positive rate
        false_count_freq = N_off / M_off if M_off > 0 else 0
        false_positive_counts = M_on * false_count_freq
        norm_false_positive_rate = (false_positive_counts / N_on) if N_on > 0 else 0
        fp_list.append([i, N_off + N_on, N_off, N_on, percent_fp_raw, norm_false_positive_rate])
        
    # Average barcodes per cell
    fake_avg = fake_barcodes.mean(axis=1)
    real_avg = real_barcodes.mean(axis=1)
    comb_avg = pd.concat([fake_avg, real_avg])
    comb_sorted = comb_avg.sort_values(ascending=False)
        
    # Create new df
    new_df = pd.DataFrame(fp_list)
    new_df.columns = ["cell name", "total_counts", "total_fake", "total_real", "FP raw", "FDR"]
    
    # Define pastel colors for blue and red with black edges
    darker_blue = "#6baed6"  # A medium blue, not too dark
    darker_red = "#fc9272"   # A soft but darker red
    edge_color = "black"
    
    # Bar plot for on and off target counts per cell with black edges
    plt.figure(figsize=(10, 6))
    x_vals = np.arange(len(new_df))
    plt.bar(x_vals, new_df["total_counts"].sort_values(ascending=False), 
            color=darker_blue, edgecolor=edge_color,linewidth=0, width=1, label="On Target")
    plt.bar(x_vals, new_df["total_fake"].sort_values(ascending=False), 
            color=darker_red, edgecolor=edge_color, linewidth=0, width=1, label="Off Target")
    plt.legend()
    plt.xlabel("Cells", fontsize=12)
    plt.ylabel("Total Counts", fontsize=12)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    sns.despine()
    plt.savefig("on_off_target_plot.svg", format="svg")
    plt.show()
    
    # Bar plot for average barcode counts per cell with black edges
    plt.figure(figsize=(10, 6))
    color = [darker_red if i.startswith("fake") else darker_blue for i in comb_sorted.index]
    plt.bar(np.arange(0, len(comb_sorted.values)), comb_sorted.values, color=color, width=1,
            edgecolor=edge_color, linewidth=0)
    plt.xlabel("Barcodes", fontsize=12)
    plt.ylabel("Average Counts per Cell", fontsize=12)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    # Add a custom legend for real and fake barcodes
    plt.legend(handles=[plt.Rectangle((0,0),1,1, color=darker_blue, edgecolor=edge_color, label="On Target"),
                        plt.Rectangle((0,0),1,1, color=darker_red, edgecolor=edge_color, label="Off Target")])
    sns.despine()
    plt.savefig("average_barcode_counts_plot.svg", format="svg")
    plt.show()

    return new_df, fake_avg

def correlation(mtx, mtx_ref, label_x=None, label_y=None, title=None, return_comb_df=False):
    """
    Output correlation plot with density coloring

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
    
    # Convert data to pseudobulk RNAseq data
    bulk = pd.DataFrame(mtx.mean(axis=1)).reset_index()
    bulk.columns = ["Genes", "Counts"]
    bulk["Genes"] = bulk["Genes"].str.lower()

    bulk_ref = pd.DataFrame(mtx_ref.mean(axis=1)).reset_index()
    bulk_ref.columns = ["Genes", "Counts ref"]
    bulk_ref["Genes"] = bulk_ref["Genes"].str.lower()

    # Merge dataframes
    comb_2 = pd.merge(bulk_ref, bulk)
    comb_2 = comb_2.drop(comb_2[comb_2["Genes"].str.startswith("fake")].index)

    # Perform linear regression
    x = comb_2["Counts ref"].values
    x_t = np.vstack([x, np.zeros(len(x))]).T
    y = comb_2["Counts"].values
    m, c = np.linalg.lstsq(x_t, y, rcond=None)[0]

    # Calculate Pearson's r
    r = pearsonr(x, y)[0]

    # Log transform the data
    comb_2["Log Counts ref"] = np.log2(comb_2["Counts ref"])
    comb_2["Log Counts"] = np.log2(comb_2["Counts"])

    # Calculate point density
    xy = np.vstack([comb_2["Log Counts ref"], comb_2["Log Counts"]])
    z = gaussian_kde(xy)(xy)  # Compute the density for each point

    # Sort the points by density, so the densest points are plotted last
    idx = z.argsort()
    x, y, z = comb_2["Log Counts ref"].values[idx], comb_2["Log Counts"].values[idx], z[idx]

    # Determine the limits with padding
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    padding = 0.1 * max(x_max - x_min, y_max - y_min)
    x_lim = (x_min - padding, x_max + padding)
    y_lim = (y_min - padding, y_max + padding)

    # Create the scatter plot with density as color
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=z, s=50, edgecolor='k', alpha=0.7, cmap="magma")

    # Add color bar for density
    cbar = plt.colorbar(scatter)
    cbar.set_label('Density', fontweight='bold', fontsize=16)

    # Create FontProperties object for bold font
    bold_font = FontProperties(weight='bold', size=12)

    # Apply bold font to color bar ticks
    cbar.ax.yaxis.set_tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(bold_font)

    # Labels and title
    plt.xlabel(f"{label_x} Log2(average counts/$\mu m^{2}$) ", fontsize=14, fontweight='bold')
    plt.ylabel(f"{label_y} Log2(average counts/$\mu m^{2}$) ", fontsize=14, fontweight='bold')
    plt.title(title, fontweight="bold")

    # Set axis ticks to bold
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Set axis limits with padding
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # Draw lines at x=0 and y=0
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
    plt.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.5)

    # Annotate in the top-left corner with bold font
    plt.annotate(f"Pearson's r= {round(r, 2)}", xy=(x_lim[0], y_lim[1]),
                 xytext=(5, -5), textcoords='offset points',
                 fontsize=16, fontweight='bold', ha='left', va='top')
    plt.annotate(f"Efficiency = {round(m, 2)}", xy=(x_lim[0], y_lim[1] - padding + 0.1),
                 xytext=(5, -10), textcoords='offset points',
                 fontsize=16, fontweight='bold', ha='left', va='top')

    # Remove the spines for a cleaner look
    sns.despine()
    
    plt.savefig(f"{label_y}_vs_smfish.svg", format="svg")

    # Show the plot
    plt.show()

    if return_comb_df:
        return comb_2