import pandas as pd
import pytest

from scripts.results_table import make_table


def test_make_table_weights_subjects_equally():
    results = pd.DataFrame(
        [
            ["Dataset", "Task", "CrossSubject", "LDA", 1, 0.6],
            ["Dataset", "Task", "CrossSubject", "LDA", 1, 0.8],
            ["Dataset", "Task", "CrossSubject", "LDA", 2, 0.4],
            ["Dataset", "Task", "CrossSubject", "SVM", 1, 0.9],
            ["Dataset", "Task", "WithinSession", "LDA", 1, 0.5],
        ],
        columns=["dataset", "paradigm", "evaluation", "pipeline", "subject", "score"],
    )

    table = make_table(results, evaluation="CrossSubject")

    # LDA: subject 1 averages to 0.7, subject 2 is 0.4.
    assert table.to_dict("records") == [
        {
            "dataset": "Dataset",
            "paradigm": "Task",
            "evaluation": "CrossSubject",
            "LDA": pytest.approx(55.0),
            "SVM": 90.0,
        }
    ]
