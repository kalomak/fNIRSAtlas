"""Canonical dataset and suite ordering metadata."""

SUITE_MAP = {
    "motor": [
        "buccino2016",
        "bak2019",
        "ortega2020",
        "luke2021",
        "cockx2023",
        "guevara2024",
        "luke2024b",
        "gao2024",
    ],
    "MA": ["bnci", "shin2016__ma", "shin2018b", "shin2019__ma"],
    "MI": ["shin2016__mi", "shin2019__mi", "yi2025"],
    "nback": ["herff2014", "shin2018__nback", "huang2021"],
    "Emotion": ["nemo__empe", "nemo__afim"],
    "MISC": ["shin2018__wg", "shader2021", "luke2024a"],
}


def get_suite_datasets(suite_name: str) -> list[str]:
    suite_name = suite_name.lower()
    for suite, datasets in SUITE_MAP.items():
        if suite.lower() == suite_name:
            return datasets
    raise ValueError(
        f"Suite '{suite_name}' not supported. Available suites: {list(SUITE_MAP.keys())}"
    )


def get_available_datasets() -> list[str]:
    return [dataset for datasets in SUITE_MAP.values() for dataset in datasets]


def get_suite_for_dataset(dataset: str):
    for suite, datasets in SUITE_MAP.items():
        if dataset in datasets:
            return suite
    return None
