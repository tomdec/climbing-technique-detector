from cv2.typing import MatLike
import matplotlib.pyplot as plt

from src.hpe.common.labels import YoloLabels
from src.hpe.common.plot import plot_average_distances
from src.hpe.mp.draw import draw_my_landmarks

def compare_landmarks(image: MatLike, labels: YoloLabels, results):
    annotated = image.copy()
    
    annotated = labels.draw(annotated)
    annotated = draw_my_landmarks(annotated, results)

    plt.imshow(annotated)
    return annotated

def plot_mediapipe_average_distances(distances, 
        save_location: str = ""):
    plot_average_distances(distances, "MediaPipe", save_location)