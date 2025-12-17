import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Optional, List, Tuple
from sklearn.metrics import ConfusionMatrixDisplay
from os.path import dirname
from os import makedirs

from src.common.kfold import AbstractFoldCrossValidation

def save_current_figure(save_location: str):
    plt.savefig(save_location, dpi=300, bbox_inches="tight")

def plot_confusion_matrix(labels: ndarray, predictions: ndarray,
        save_path: Optional[str] = None,
        normalized=False):
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=labels, 
        y_pred=predictions, 
        normalize="true" if normalized else None,
        xticks_rotation=45,
        cmap="Blues", 
        values_format="0.2f" if normalized else "g",
        text_kw={'size': 10})
    
    if normalized:
        disp.ax_.set_title("Confusion Matrix Normalized")
    else:
        disp.ax_.set_title("Confusion Matrix")

    if save_path:
        makedirs(dirname(save_path), exist_ok=True)
        save_current_figure(save_path)
    else:
        plt.show()

def box_plot_accuracies(kfold_models: List[AbstractFoldCrossValidation],
        save_path: str | None = None):
    
    def get_name(fold_model: AbstractFoldCrossValidation) -> str:
        return fold_model._model_args.name
    
    def get_metrics(model: AbstractFoldCrossValidation) -> List[float]:
        return model.get_test_accuracy_metrics()

    metrics = list(map(get_metrics, kfold_models))
    names = list(map(get_name, kfold_models))
    
    plt.figure()
    plt.title("Comparison of test accuracies")
    plt.boxplot(metrics, tick_labels=names)
    plt.ylabel("Test accuracy [%]")
    
    if save_path:
        save_current_figure(save_path)
    else:
        plt.show()
    

def plot_histograms(names: List[str],
        data: List[List[float]],
        save_location: str = "",
        title: str = "",
        legend_location: str = "upper right",
        xlabel: str = "",
        xlim: Tuple[float, float] | None = None,
        weights: List[List[float]] | None = None,
        logscale: bool = False):
    
    if weights:
        for name, arr, weight_arr in zip(names, data, weights):
            plt.hist(arr, bins=100, label=name, weights=weight_arr)
    else:
        for name, arr in zip(names, data):
            plt.hist(arr, bins=100, label=name)

    plt.legend(loc=legend_location)
    
    if title: plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    if xlim: plt.xlim(xlim)
    if logscale:
        plt.xscale('log')
        plt.yscale('log')

    if save_location: save_current_figure(save_location)

def plot_histogram_grid(names: List[str],
        data: List[List[float]],
        grid: Tuple[int, int],
        save_location: str = "",
        title: str = "",
        xlabel: str = ""):
    
    if len(names) != grid[0] * grid[1]:
        raise Exception("Should provide an equal amount of data as requested grids.")

    fig, axes = plt.subplots(grid[0], grid[1], sharex="col")
    axes: List[plt.Axes] = axes.reshape(-1)    
    for name, arr, ax, idx in zip(names, data, axes, range(len(names))):
        ax.hist(arr, bins=[x / 100 for x in range(100)], label=name)
        
        ax.set_title(name)
        if xlabel and idx == len(data)-1: 
            ax.set_xlabel(xlabel)

    if save_location: save_current_figure(save_location)