import pytest

import src.labels as labels

def pytest_configure(config):
    """
    Mock the __labels variable, so that the test are independent from the actual /labels.yml file. 
    """
    test_labels = {
            "name": "test-techniques",
            "values": [
                "INVALID", 
                "NONE", 
                "FOOT_SWAP",
                "OUTSIDE_FLAG",
                "BACK_FLAG",
                "INSIDE_FLAG",
                "DROP_KNEE",
                "CROSS_MIDLINE"
            ]
        }
    labels.__labels = test_labels