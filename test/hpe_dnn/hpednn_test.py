import pytest

from src.hpe_dnn.architecture import DnnArch
from src.hpe_dnn.model import HpeDnn, HpeDnnConstructorArgs, HpeDnnModelInitializeArgs


def test_HpeDnn_get_best_model_path():
    dnn_model = HpeDnn(
        args=HpeDnnConstructorArgs(
            name="test_model",
            model_initialize_args=HpeDnnModelInitializeArgs(model_arch=DnnArch.ARCH2),
            data_root_path="test/data",
        )
    )
    expected = "test/data/runs/hpe_dnn/test_model/train5/models/epoch_01__val_accuracy_0.9414.keras"

    actual = dnn_model._get_best_model_path()

    assert actual == expected
