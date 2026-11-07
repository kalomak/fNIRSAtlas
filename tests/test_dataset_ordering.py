#!/usr/bin/env python3
"""
Test script for dataset ordering functionality
"""

import pytest
from fnirsatlas.parsers import get_suite_datasets


def test_get_suite_datasets():
    """Test that get_suite_datasets returns datasets in correct order."""
    motor_datasets = get_suite_datasets("motor")

    # Check that datasets are returned
    assert isinstance(motor_datasets, list)
    assert len(motor_datasets) > 0

    # Check that motor datasets are in correct order
    expected_motor = [
        "buccino2016",
        "bak2019",
        "ortega2020",
        "luke2021",
        "cockx2023",
        "guevara2024",
        "luke2024b",
        "gao2024",
    ]
    assert motor_datasets == expected_motor


def test_get_suite_datasets_with_suite():
    """Test that get_suite_datasets with suite parameter works."""
    motor_datasets = get_suite_datasets("motor")
    assert isinstance(motor_datasets, list)
    assert len(motor_datasets) > 0
    assert all(
        dataset
        in [
            "buccino2016",
            "bak2019",
            "ortega2020",
            "luke2021",
            "cockx2023",
            "guevara2024",
            "luke2024b",
            "gao2024",
        ]
        for dataset in motor_datasets
    )


def test_consistent_ordering():
    """Test that ordering is consistent between calls."""
    motor1 = get_suite_datasets("motor")
    motor2 = get_suite_datasets("motor")

    assert motor1 == motor2


if __name__ == "__main__":
    # Run tests
    test_get_suite_datasets()
    test_get_suite_datasets_with_suite()
    test_consistent_ordering()
    print("All dataset ordering tests passed!")
