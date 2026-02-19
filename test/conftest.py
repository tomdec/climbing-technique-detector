import pytest

import src.labels as labels


def pytest_configure(config):
    """
    Mock the __labels variable, so that the test are independent from the actual /labels.yml file.
    """
    test_labels = {
        "name": "test-labels",
        "values": [
            "INVALID",
            "LABEL1",
            "LABEL2",
            "LABEL3",
            "LABEL4",
            "LABEL5",
            "LABEL6",
            "LABEL7",
        ],
    }
    labels.__labels = test_labels
