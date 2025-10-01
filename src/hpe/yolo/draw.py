from cv2.typing import MatLike
from ultralytics.engine.results import Results
from supervision import VertexAnnotator, Color, KeyPoints as sv_Keypoints

from src.hpe.yolo.landmarks import PredictedLandmarks

def draw_my_landmarks(image: MatLike, results: PredictedLandmarks) -> MatLike:
    annotated = image.copy()
    sv_keypoints = sv_Keypoints.from_ultralytics(results.values)

    vertex_annotator = VertexAnnotator(
        radius=5, 
        color=Color.WHITE)

    annotated = vertex_annotator.annotate(
        scene=annotated,
        key_points=sv_keypoints)

    return annotated