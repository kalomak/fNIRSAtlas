"""Compare fNIRS pipelines with MOABB's statistical analysis."""

import matplotlib.pyplot as plt
from moabb.analysis import compute_dataset_statistics, find_significant_differences
from moabb.analysis.plotting import summary_plot
from moabb.evaluations import CrossSubjectEvaluation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fnirsatlas import FingerTapping
from fnirsatlas.pipelines import WindowedFeatureExtractor

paradigm = FingerTapping()
evaluation = CrossSubjectEvaluation(paradigm=paradigm, datasets=paradigm.datasets)
results = evaluation.process(
    {
        "Windowed + LDA": make_pipeline(
            WindowedFeatureExtractor(),
            StandardScaler(),
            LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
        ),
        "Windowed + SVM-RBF": make_pipeline(
            WindowedFeatureExtractor(), StandardScaler(), SVC(kernel="rbf")
        ),
    }
)

stats = compute_dataset_statistics(results)
p_values, effects = find_significant_differences(stats)
print("Effects:\n", effects)
print("P-values:\n", p_values)
summary_plot(p_values, effects)
plt.show()
