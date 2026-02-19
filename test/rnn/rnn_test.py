import pytest

from src.rnn.architecture import RnnArch
from src.rnn.model import Rnn, RnnConstructorArgs, RnnModelInitializeArgs


@pytest.mark.parametrize(
    "model_name,expected",
    [
        (
            "test_model",
            "test/data/runs/rnn/test_model/train2/models/epoch_02__val_accuracy_0.9597.keras",
        ),
        (
            "test_model2",
            "test/data/runs/rnn/test_model2/train/models/epoch_04__val_accuracy_0.9597.keras",
        ),
    ],
)
def test_Rnn_get_best_model_path(model_name, expected):

    sota_model = Rnn(
        args=RnnConstructorArgs(
            name=model_name,
            model_initialize_args=RnnModelInitializeArgs(model_arch=RnnArch.ARCH1),
            data_root_path="test/data",
        )
    )

    actual = sota_model._get_best_model_path()

    assert actual == expected
