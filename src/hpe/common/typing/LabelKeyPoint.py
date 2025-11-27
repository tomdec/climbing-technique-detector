from numpy import array, ndarray
from typing import override

class LabelKeyPoint(DrawableKeyPoint):

    @staticmethod
    def from_dict(values: dict, name: str) -> 'LabelKeyPoint':
        return LabelKeyPoint(
            x=values['x'],
            y=values['y'],
            visibility=values['visibility'],
            name=name)

    _visibility: Visibility

    def __init__(self, x, y, visibility, name: str | None = None):
        DrawableKeyPoint.__init__(self, x=float(x), y=float(y),
            name="" if name is None else name)
        self._visibility = visibility\
            if type(visibility) is Visibility\
            else Visibility(int(visibility))

    def __str__(self):
        return f"{self.as_dict()}"

    @override
    def is_missing(self) -> bool:
        return self._visibility == Visibility.MISSING

    def as_array(self) -> ndarray:
        return array([self._x, self._y])

    def as_dict(self) -> dict:
        return {
            'x': self._x,
            'y': self._y,
            'visibility': self._visibility,
            #'name': self._name
        }