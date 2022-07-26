"""
author: Katsuya Lex Colon
group: CaiLab
date:02/25/22
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ecdf:    
    def values(data, label):
        """Calculates ecdf from a 1D array
        Parameters
        ----------
        data = 1D array
        label = string
        Returns 
        Pandas dataframe with ecdf values, labels, and x values
        Returns 
        -------
        ecdf values
        """

        data = data[~np.isnan(data)]
        ecdf = pd.DataFrame(np.arange(1, len(data)+1) / len(data))
        ecdf['Label'] = label
        ecdf['Values'] = np.sort(data)
        ecdf.columns = ["ecdf", "Label", "Values"]

        return ecdf

    def plot(data, label_column, val_column, conf = False, color = None):
        """generates ecdf plots
        Parameters
        ----------
        data = tidy data with values in one column and labels in another
        label_column = column name with categories
        value_column = column name with values
        color = list
        conf = 95% confidence interval generated from bootstrap resampling
        Returns 
        -------
        editable ecdf plot using matplotlib
        """

        #obtain unique values
        labels = data[label_column].unique()

        #calculate ecdf for every label
        ecdf_list = []
        for i in labels:
            sliced_data = data.loc[data[label_column] == i]
            ecdf_val = ecdf.values(sliced_data[val_column].values, i)
            ecdf_list.append(ecdf_val)

        #plot ecdf
        for i in range(len(ecdf_list)):
            try: 
                plt.step(ecdf_list[i]["Values"], ecdf_list[i]["ecdf"], linewidth = 1.5, 
                         label = ecdf_list[i]["Label"][0], color = color[i])
            except TypeError:
                plt.step(ecdf_list[i]["Values"], ecdf_list[i]["ecdf"], linewidth = 1.5, 
                         label = ecdf_list[i]["Label"][0])
            
        #plot conf interval
        if conf == True:
            if color == None:
                print("\x1b[31m\"Error CI: Specify color scheme by supplying a list of colors in the color argument to match the ecdf plot\"\x1b[0m")   
            else:
                #slice by labels
                k = 0
                for i in labels:
                    sliced_data = data.loc[data[label_column] == i]
                    #initialize array with 1 bootstrap sample
                    init_arr = np.sort(np.random.choice(sliced_data[val_column].values, 
                                                        size=len(sliced_data[val_column].values)))
                    
                    #obtain the rest of bootstrap samples
                    for _ in range(2000):
                        x = np.sort(np.random.choice(sliced_data[val_column].values, 
                                                     size=len(sliced_data[val_column].values)))
                        init_arr = np.vstack((init_arr, x))

                    #ecdf for all bs sample
                    list_ecdf = []
                    for i in init_arr:
                        ecdf_dict = dict(zip(np.sort(i), np.arange(1, len(i)+1) / len(i)))
                        list_ecdf.append(ecdf_dict)

                    #pool all ecdf for a given value and calculate CI
                    ci_list = []
                    for i in np.sort(sliced_data[val_column].values):
                        list_values = []
                        for j in list_ecdf:
                            try: 
                                ecdf_i = j[i]
                                list_values.append(ecdf_i)
                            except KeyError:
                                continue
                        ecdf_i_arr = np.asarray(list_values)
                        low, upp = np.percentile(ecdf_i_arr, [2.5, 97.5])
                        ci_list.append((i, low, upp))

                    #convert ci to dataframe    
                    conf_df = pd.DataFrame(ci_list)
                    conf_df.columns = ["x", "lower bound", "upper bound"]

                    #plot bands
                    plt.fill_between(conf_df["x"], conf_df["lower bound"], conf_df["upper bound"], 
                                     alpha = 0.3, color = color[k], step = "pre")
                    k += 1
                
        plt.legend()
        sns.despine()