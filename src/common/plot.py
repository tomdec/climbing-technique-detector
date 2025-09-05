import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Optional, List
from sklearn.metrics import ConfusionMatrixDisplay
from os.path import dirname
from os import makedirs

from src.common.kfold import AbstractFoldCrossValidation
from src.labels import iterate_valid_labels

__TICKS = [label for label in iterate_valid_labels()]

def plot_confusion_matrix(labels: ndarray, predictions: ndarray,
        save_path: Optional[str] = None,
        normalized=False):
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true=labels, 
        y_pred=predictions, 
        normalize="true" if normalized else None,
        display_labels=__TICKS,
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
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

def box_plot_accuracies(*kfold_models: List[AbstractFoldCrossValidation]):
    
    def get_name(fold_model: AbstractFoldCrossValidation) -> str:
        return fold_model._model_args.name
    
    def get_metrics(model: AbstractFoldCrossValidation) -> List[float]:
        return model.get_test_accuracy_metrics()

    metrics = list(map(get_metrics, kfold_models))
    names = list(map(get_name, kfold_models))
    
    plt.figure()
    plt.title("Comparison of test accuracies")
    plt.boxplot(metrics, tick_labels=names)
    plt.show()
