from enum import Enum
from typing import Mapping, Callable
from keras.api.models import Sequential
from keras.api.layers import (
    InputLayer,
    ConvLSTM2D,
    MaxPooling2D,
    Conv2D,
    Dense,
    Reshape,
    Flatten,
)
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy
from keras.api.metrics import CategoricalAccuracy

from src.labels import get_valid_label_count
from src.sampling.images import FRAME_SIZE


class ConvLstmArch(Enum):
    ARCH1 = 1


def __arch1_factory() -> Sequential:
    model = Sequential(
        [
            InputLayer(shape=(5, FRAME_SIZE, FRAME_SIZE, 3)),
            ConvLSTM2D(32, 3, return_sequences=False),
            MaxPooling2D(2),
            Conv2D(32, 3),
            MaxPooling2D(2),
            Conv2D(32, 3),
            MaxPooling2D(2),
            Conv2D(32, 3),
            MaxPooling2D(2),
            Flatten(),
            Reshape((1, -1)),
            Dense(get_valid_label_count(), activation="softmax"),
        ]
    )
    return model


__arch_mapping: Mapping[ConvLstmArch, Callable[[], Sequential]] = {
    ConvLstmArch.ARCH1: __arch1_factory,
}


def get_model(architecture: ConvLstmArch, loss=CategoricalCrossentropy()) -> Sequential:
    factory = __arch_mapping[architecture]
    model = factory()
    model.compile(loss=loss, optimizer=Adam(), metrics=[CategoricalAccuracy()])
    return model
