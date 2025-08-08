import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
from numpy import ndarray, argmax, transpose
from typing import Optional
from pandas import DataFrame
from numpy import transpose
from os.path import dirname
from os import makedirs

__TICKS = ["NONE", "FOOT_SWAP", "OUTSIDE_FLAG", "BACK_FLAG", "INSIDE_FLAG", "DROP_KNEE", "CROSS_MIDLINE"]

def plot_confusion_matrix(labels: ndarray, predictions: ndarray,
        save_path: Optional[str] = None):
    
    label_idx = argmax(labels, axis=1)
    pred_idx = argmax(predictions, axis=1)
    
    cm = tf.math.confusion_matrix(labels=label_idx, predictions=pred_idx)
    cm = transpose(cm)
    df_cm = DataFrame(cm, index=__TICKS, columns=__TICKS)
    
    fig = plt.figure(figsize=(8, 5))
    sn.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 10}, fmt="g")

    plt.title("Confusion Matrix")
    plt.xlabel("True")
    plt.ylabel("Predicted")

    if save_path:
        makedirs(dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()