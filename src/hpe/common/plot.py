from typing import List, Tuple
from matplotlib import pyplot as plt
from pandas import DataFrame
from numpy import ones, average, nan, isnan
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
        
def plot_distances_boxplot(
        ylim: Tuple[int, int] | None  = None,
        *distances: List[Tuple[str, DataFrame]]):
    
    def get_name(distance: Tuple[str, DataFrame]) -> str:
        total = distance[1].shape[0] * distance[1].shape[1]
        missing = 1 - len(get_values(distance)) / total
        return f"{distance[0]}\nmissing: {missing*100:4.2f}%"
    
    def get_values(distance: Tuple[str, DataFrame]) -> List[float]:
        values = distance[1].values.reshape((-1))
        return [value for value in values if not isnan(value)]
    
    metrics = list(map(get_values, distances))
    names = list(map(get_name, distances))
    
    figure = plt.figure()
    plt.title("Comparison of HPE distances")
    plt.boxplot(metrics, tick_labels=names)
    if ylim is not None:
        figure.axes[0].set_ylim(ylim)
    plt.show()