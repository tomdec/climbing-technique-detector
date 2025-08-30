from cv2.typing import MatLike
from ultralytics.engine.results import Results, Keypoints
import matplotlib.pyplot as plt
from numpy import ones

from src.hpe.common.labels import YoloLabels, MyLandmark
from src.hpe.common.plot import plot_average_distances
from src.hpe.yolo.landmarks import get_pose_landmark
from src.hpe.yolo.draw import draw_my_landmarks

def compare_landmarks(image: MatLike, 
        labels: YoloLabels,
        results: Results,
        object_index: int = 0) -> MatLike:
    return compare_landmarks(image, labels, results[object_index].keypoints)

def compare_landmarks(image: MatLike, 
        labels: YoloLabels,
        results: Keypoints) -> MatLike:
    annotated = image.copy()
    
    annotated = labels.draw(annotated)
    annotated = draw_my_landmarks(annotated, results)

    plt.imshow(annotated)
    return annotated

def plot_distances(distances):
    """Plot distances from predictions on a single image.

    Args:
        distances (Any): Distances between true labels and predicted landmarks.
    """
    limit = ones(37)
    plt.figure()
    plt.title(f"Yolo prediction distances")
    plt.plot(limit, color='red')

    for landmark in MyLandmark:
        if (get_pose_landmark(landmark) is not None):
            index = landmark.value
            plt.plot(distances.T[index], label = f"{landmark.name}")

    plt.legend(loc="upper right")
    plt.ylim(0, 10)
    plt.xlim(0, 60)


def plot_yolo_average_distances(distances, 
        save_location: str = ""):
    plot_average_distances(distances, "yolo11x-pose", save_location)