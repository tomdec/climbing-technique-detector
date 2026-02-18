from enum import Enum
from typing import Mapping, Callable
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Reshape, Dropout
from keras.api.optimizers import Adam
from keras.api.losses import CategoricalCrossentropy
from keras.api.metrics import CategoricalAccuracy

from src.labels import get_valid_label_count


class RnnArch(Enum):
    DNNMIMIC = -2
    ARCH0 = -1
    ARCH1 = 0
    ARCH2 = 1
    ARCH3 = 2
    ARCH4 = 3
    ARCH5 = 4
    ARCH6 = 5


def __control() -> Sequential:
    lstm_model = Sequential(
        [
            LSTM(1, return_sequences=False),
            Dense(units=get_valid_label_count(), activation="softmax"),
            Reshape((-1, get_valid_label_count())),
        ]
    )
    return lstm_model


def __arch1_factory() -> Sequential:
    lstm_model = Sequential(
        [
            LSTM(256, return_sequences=True),
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Reshape((-1, 32)),
            Dense(units=get_valid_label_count(), activation="softmax"),
        ]
    )
    return lstm_model


def __arch2_factory() -> Sequential:
    lstm_model = Sequential(
        [
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Reshape((-1, 32)),
            Dense(units=get_valid_label_count(), activation="softmax"),
        ]
    )
    return lstm_model


def __arch3_factory() -> Sequential:
    lstm_model = Sequential(
        [
            LSTM(1024, return_sequences=True),
            LSTM(512, return_sequences=True),
            LSTM(256, return_sequences=True),
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Reshape((-1, 32)),
            Dense(units=get_valid_label_count(), activation="softmax"),
        ]
    )
    return lstm_model


def __arch4_factory() -> Sequential:

    lstm_model = Sequential(
        [
            Dense(512, activation="relu"),
            Dropout(rate=0.1),
            Dense(256, activation="relu"),
            Dropout(rate=0.1),
            Dense(128, activation="relu"),
            Dropout(rate=0.1),
            Dense(64, activation="relu"),
            Dropout(rate=0.1),
            LSTM(32, return_sequences=False),
            Reshape((-1, 32)),
            Dense(units=get_valid_label_count(), activation="softmax"),
        ]
    )
    return lstm_model


def __arch5_factory() -> Sequential:

    lstm_model = Sequential(
        [
            LSTM(512, return_sequences=False),
            Reshape((-1, 512)),
            Dense(256, activation="relu"),
            Dropout(rate=0.1),
            Dense(128, activation="relu"),
            Dropout(rate=0.1),
            Dense(64, activation="relu"),
            Dropout(rate=0.1),
            Dense(32, activation="relu"),
            Dropout(rate=0.1),
            Dense(units=get_valid_label_count(), activation="softmax"),
        ]
    )
    return lstm_model


def __arch6_factory() -> Sequential:

    lstm_model = Sequential(
        [
            Dense(512, activation="relu"),
            Dropout(rate=0.1),
            Dense(256, activation="relu"),
            Dropout(rate=0.1),
            Dense(128, activation="relu"),
            Dropout(rate=0.1),
            Dense(64, activation="relu"),
            Dropout(rate=0.1),
            Dense(32, activation="relu"),
            Dropout(rate=0.1),
            LSTM(32, return_sequences=True),
            LSTM(32, return_sequences=True),
            LSTM(32, return_sequences=True),
            LSTM(32, return_sequences=False),
            Reshape((-1, 32)),
            Dense(units=get_valid_label_count(), activation="softmax"),
            Reshape((-1, get_valid_label_count())),
        ]
    )
    return lstm_model


def __hpe_dnn_mimic_factory() -> Sequential:

    lstm_model = Sequential(
        [
            Dense(512, activation="relu"),
            Dropout(rate=0.1),
            Dense(256, activation="relu"),
            Dropout(rate=0.1),
            Dense(128, activation="relu"),
            Dropout(rate=0.1),
            Dense(64, activation="relu"),
            Dropout(rate=0.1),
            Dense(32, activation="relu"),
            Dropout(rate=0.1),
            Dense(units=get_valid_label_count(), activation="softmax"),
            Reshape((-1, get_valid_label_count())),
        ]
    )
    return lstm_model


__arch_mapping: Mapping[RnnArch, Callable[[], Sequential]] = {
    RnnArch.DNNMIMIC: __hpe_dnn_mimic_factory,
    RnnArch.ARCH0: __control,
    RnnArch.ARCH1: __arch1_factory,
    RnnArch.ARCH2: __arch2_factory,
    RnnArch.ARCH3: __arch3_factory,
    RnnArch.ARCH4: __arch4_factory,
    RnnArch.ARCH5: __arch5_factory,
    RnnArch.ARCH6: __arch6_factory,
}


def get_model(architecture: RnnArch, loss=CategoricalCrossentropy()) -> Sequential:
    factory = __arch_mapping[architecture]
    model = factory()
    model.compile(
        loss=loss,
        optimizer=Adam(),
        metrics=[CategoricalAccuracy()],
    )

    return model
