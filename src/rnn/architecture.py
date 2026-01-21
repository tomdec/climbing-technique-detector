from enum import Enum
from typing import Mapping, Callable
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Reshape
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy
from keras.api.metrics import CategoricalAccuracy

from src.labels import get_valid_label_count


class RnnArch(Enum):
    ARCH1 = 0


def __arch1_factory() -> Sequential:
    lstm_model = Sequential(
        [
            LSTM(128, return_sequences=False),
            Dense(units=get_valid_label_count(), activation="softmax"),
            Reshape((-1, get_valid_label_count())),
        ]
    )
    return lstm_model


__arch_mapping: Mapping[RnnArch, Callable[[], Sequential]] = {
    RnnArch.ARCH1: __arch1_factory
}


def get_model(architecture: RnnArch) -> Sequential:
    factory = __arch_mapping[architecture]
    model = factory()
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=[CategoricalAccuracy()],
    )

    return model
