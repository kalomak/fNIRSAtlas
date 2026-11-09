"""
Export preprocessed fNIRS data
==============================

Load preprocessed data and use it outside MOABB evaluations.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification, WindowedFeatureExtractor

dataset = Bnci2014FnirsMA(subjects=[1])
paradigm = FNIRSClassification()
X, y, metadata = paradigm.get_data(dataset, subjects=[1])

model = make_pipeline(
    WindowedFeatureExtractor(),
    StandardScaler(),
    LinearDiscriminantAnalysis(),
)
model.fit(X, y)
print(X.shape, metadata.shape)
print(model.predict(X[:5]))
