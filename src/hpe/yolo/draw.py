from cv2.typing import MatLike
from ultralytics.engine.results import Keypoints
from supervision import VertexAnnotator, Color, KeyPoints as sv_Keypoints

def draw_my_landmarks(image: MatLike, results: Keypoints) -> MatLike:
    annotated = image.copy()
    sv_keypoints = sv_Keypoints.from_ultralytics(results)

    vertex_annotator = VertexAnnotator(
        radius=15, 
        color=Color.WHITE)

    annotated = vertex_annotator.annotate(
        scene=annotated,
        key_points=sv_keypoints)

    return annotated