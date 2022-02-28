from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dot_displacement(ref, moving, radius=2):
    """
    This function will calculate the displacement of dots by fitting a 1d gaussian
    to a distance array obtained from colocalizing dots. The full width half maximum of this
    1D gaussian will correspond to displacement.
    
    Parameters
    ----------
    ref: the reference dots
    moving: the aligned dots
    radius: search radius for colocalization
    
    Return
    ----------
    average distance off, displacement 
    """
    #reset index for df just in case
    ref = ref.reset_index(drop=True)
    moving = moving.reset_index(drop=True)
    
    #using sklearn nearest neighbor algorithm to find nearest dots
    #initialize algorithm
    neigh = NearestNeighbors(n_neighbors=2, radius=radius, metric="euclidean", n_jobs=1)
    
    #initialize neighbor
    initial_seed = ref[["x","y"]]
    #find neighbors for df1
    neigh.fit(moving[["x","y"]])
    distances,_ = neigh.radius_neighbors(initial_seed, radius, return_distance=True, sort_results=True)
    
    #distances flattened
    distances_flattened = []
    for i in range(len(distances)):
        try:
            distances_flattened.append([distances[i][0]])
        except IndexError:
            continue
            
    #make 1d array
    distances_arr = np.array(distances_flattened).ravel()
    
    #get negative values
    distances_neg = -(np.array(distances_flattened).ravel())
    
    #combine
    distances_arr = np.concatenate([distances_neg,distances_arr])
    
    #fit gaussian distribution
    mu, std = norm.fit(distances_arr) 
    xmin, xmax = min(distances_arr), max(distances_arr)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    #get half maximum
    half_max = max(p)/2
    
    #get half width at half maximum
    index_hwhm_1 = np.where(p > max(p)/2)[0][-1]
    
    #get displacement by looking at fullwidth
    displacement = x[index_hwhm_1]*2
    
    #plot histogram
    plt.hist(distances_arr, density=True, bins=15)
    #plot distribution
    plt.plot(x,p, label="Gaussian Fitted Data")
    #plot half max
    plt.axhline(half_max, color="red")
    #plot full width
    plt.axvline(displacement/2, color="red", label="FWHM")
    plt.axvline(-displacement/2, color="red")
    plt.legend()
    sns.despine()
    plt.ylabel("Probability density")
    plt.xlabel("Relative distances (pixels)")
    plt.show()
    
    return displacement
