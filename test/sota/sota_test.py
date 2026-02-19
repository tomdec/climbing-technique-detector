import pytest

from src.sota.model import SOTA, SOTAConstructorArgs, SOTAModelInitializeArgs


def test_SOTA_get_best_model_path():
    sota_model = SOTA(
        args=SOTAConstructorArgs(
            name="test_model",
            model_initialize_args=SOTAModelInitializeArgs(model_arch="yolo11n-cls"),
            data_root_path="test/data",
        )
    )
    expected = "test/data/runs/sota/test_model/train5/weights/best.pt"

    actual = sota_model._get_best_model_path()

    assert actual == expected
