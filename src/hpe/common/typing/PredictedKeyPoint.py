from numpy import array, ndarray

from src.hpe.common.typing.DrawableKeyPoint import DrawableKeyPoint

class PredictedKeyPoint(DrawableKeyPoint):

    @staticmethod
    def from_dict(values: dict, name: str) -> 'PredictedKeyPoint':
        return PredictedKeyPoint(
            x=values['x'],
            y=values['y'],
            z=values['z'],
            visibility=values['visibility'],
            name=name
        )

    @staticmethod
    def empty() -> 'PredictedKeyPoint':
        """
        To be used when a landmark is missing because the person (or body part) is not detected
        """
        return PredictedKeyPoint(x=0.0, y=0.0, z=None, visibility=1, name="")

    @property
    def z(self) -> float | None:
        """Estimated z-coordinate (or depth) of the landmark. Uses scaling of x-axis."""
        return self._z

    @property
    def visibility(self) -> float:
        """Likelihood of the landmark being visible, in range [0.0, 1.0]."""
        return self._visibility

    def __init__(self,
            x: float,
            y: float,
            z: float | None,
            visibility: float,
            name: str):
        DrawableKeyPoint.__init__(self, x, y, name)
        self._z = z
        self._visibility = visibility

    def __str__(self) -> str:
        return f"{self.as_dict()}"

    def as_array(self) -> ndarray:
        return array([self._x, self._y])

    def as_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility,
            #'name': self.name
        }