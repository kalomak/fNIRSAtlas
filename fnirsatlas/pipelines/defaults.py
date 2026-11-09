"""Default pipelines for fNIRS benchmarks."""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .transformations import WindowedFeatureExtractor


def make_fnirs_lda_pipeline(n_windows=6, features=None, pca_variance=None):
    """Build the windowed-feature shrinkage-LDA pipeline.

    Parameters
    ----------
    n_windows : int
        Number of temporal windows used for feature extraction.
    features : list of str or None
        Features passed to :class:`WindowedFeatureExtractor`.
    pca_variance : float or None
        Fraction of variance kept by a PCA step placed before LDA. ``None``
        fits LDA on the full feature space, whose shrinkage covariance costs
        O(n_features**2) memory and O(n_features**3) time.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Windowed features, standardization, optional PCA, then shrinkage LDA.
    """
    steps = [
        WindowedFeatureExtractor(n_windows=n_windows, features=features),
        StandardScaler(),
    ]
    if pca_variance is not None:
        steps.append(PCA(n_components=pca_variance))
    steps.append(LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"))
    return make_pipeline(*steps)


def make_fnirs_svm_pipeline(
    n_windows=6, features=None, kernel="rbf", C=1.0, gamma="scale"
):
    """Build the default windowed-feature SVM pipeline.

    Parameters
    ----------
    n_windows : int
        Number of temporal windows used for feature extraction.
    features : list of str or None
        Features passed to :class:`WindowedFeatureExtractor`.
    kernel : str
        SVM kernel, for example ``"rbf"`` or ``"linear"``.
    C : float
        Regularization strength.
    gamma : str or float
        Kernel coefficient, ignored for a linear kernel.
    """
    return make_pipeline(
        WindowedFeatureExtractor(n_windows=n_windows, features=features),
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma),
    )
