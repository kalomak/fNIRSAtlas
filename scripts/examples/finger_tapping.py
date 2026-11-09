"""Run finger-tapping classification on every compatible dataset."""

from moabb.evaluations import CrossSubjectEvaluation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import FingerTapping
from fnirsatlas.pipelines import WindowedFeatureExtractor

paradigm = FingerTapping()
datasets = paradigm.datasets
print([dataset.code for dataset in datasets])

evaluation = CrossSubjectEvaluation(paradigm=paradigm, datasets=datasets)
results = evaluation.process(
    {
        "Windowed + LDA": make_pipeline(
            WindowedFeatureExtractor(),
            StandardScaler(),
            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        )
    }
)
print(results.groupby("dataset")["score"].mean())
