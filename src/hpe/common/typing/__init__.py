from typing import Dict

from src.hpe.common.typing.MyLandmark import MyLandmark
from src.hpe.common.typing.DrawableKeyPoint import DrawableKeyPoint
from src.hpe.common.typing.Visibility import Visibility
from src.hpe.common.typing.PredictedKeyPoint import PredictedKeyPoint
from src.hpe.common.typing.LabelKeyPoint import LabelKeyPoint
from src.hpe.common.typing.KeypointDrawConfig import KeypointDrawConfig
from src.hpe.common.typing.HpeEstimation import HpeEstimation

PerformanceMap = Dict[MyLandmark, bool | None]
"""
Dictionary that maps each value of MyLandmark to either:
- True: the landmark was correctly detected
- False: the landmark was not (correctly) detected
- None: the tool cannot detect the landmark
"""