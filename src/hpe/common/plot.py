from matplotlib import pyplot as plt
from numpy import ones, average, isnan, nan
from numpy.ma import masked_array

def plot_average_distances(distances,
        title: str,
        save_location: str = ""):
    
    limit = ones(37)

    plt.figure()
    plt.title(title)
    plt.plot(limit, color='red', label="PCKh50 limit")
    plt.xlabel("images")
    plt.ylabel("average relative error")
    plt.legend(loc="upper right")

    masked_distances = masked_array(distances, isnan(distances))
    masked_dist_av = average(masked_distances, 1)
    dist_av = masked_dist_av.filled(nan)

    plt.bar(range(0, 37), dist_av)

    if save_location != "":
        plt.savefig(save_location)
        
def plot_distances_boxplot(distances1, distances2):
    pass