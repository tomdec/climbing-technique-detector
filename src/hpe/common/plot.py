from typing import List, Tuple
from matplotlib import pyplot as plt
from pandas import DataFrame
from numpy import ones, average, nan, isnan
from numpy.ma import masked_array

from src.hpe.common.landmarks import MyLandmark

def plot_average_distances(distances: DataFrame,
        title: str,
        save_location: str = ""):
    x_width = distances.shape[0]
    limit = ones(x_width)

    plt.figure()
    plt.title(title)
    plt.plot(limit, color='red', label="PCKh50 limit")
    plt.xlabel("images")
    plt.ylabel("average relative error")
    plt.legend(loc="upper right")

    masked_distances = masked_array(distances, isnan(distances))
    masked_dist_av = average(masked_distances, 1)
    dist_av = masked_dist_av.filled(nan)

    plt.bar(range(0, x_width), dist_av)

    if save_location != "":
        plt.savefig(save_location)
        
def plot_distances_boxplot(
        ylim: Tuple[int, int] | None  = None,
        save_location: str = "",
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

    if save_location != "":
        plt.savefig(save_location)

def plot_precision_recall_curve(pnr: DataFrame, 
        tight: bool = True, 
        columns=None, 
        save_location: str = ""):
    """Plot the precision-recall curves for each class in the DataFrame. 

    Args:
        pnr (DataFrame): DataFrame with 
        tight (bool, optional): _description_. Defaults to True.
        columns (_type_, optional): _description_. Defaults to None.
    """
    if columns is None:
        columns = [landmark for landmark in MyLandmark]
    
    for landmark in columns:
        plt.plot(
            pnr[landmark.name].map(lambda x: x['r'])[:-1], 
            pnr[landmark.name].map(lambda x: x['p'])[:-1])
        
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    #plt.legend([landmark.name for landmark in MyLandmark])

    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

    if save_location != "":
        plt.savefig(save_location)

def plot_precision_and_recall(pnr: DataFrame, tight: bool = True):
    
    plt.figure()
    plt.subplot(121)
    for landmark in MyLandmark:
        plt.plot(
            pnr["CONFIDENCE"],
            pnr[landmark.name].map(lambda x: x['p']))
        
    plt.title("Precision")
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    #plt.legend([landmark.name for landmark in MyLandmark])

    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)

    plt.subplot(122)
    for landmark in MyLandmark:
        plt.plot(
            pnr["CONFIDENCE"],
            pnr[landmark.name].map(lambda x: x['r']))
        
    plt.title("Recall")
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    
    if not tight:
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)