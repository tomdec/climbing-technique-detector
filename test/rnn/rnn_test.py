from src.rnn.architecture import RnnArch
from src.rnn.model import Rnn, RnnConstructorArgs, RnnModelInitializeArgs


def test_Rnn_get_best_model_path():

    sota_model = Rnn(
        args=RnnConstructorArgs(
            name="test_model",
            model_initialize_args=RnnModelInitializeArgs(model_arch=RnnArch.ARCH1),
            data_root_path="test/data",
            dataset_name="techniques",
        )
    )
    expected = "test/data/runs/rnn/test_model/train2/models/epoch_02__val_accuracy_0.9597.keras"

    actual = sota_model._get_best_model_path()

    assert actual == expected
