"""Shared preprocessing for chromophore-split neural classifiers."""

from sklearn.pipeline import make_pipeline

from fnirsatlas.pipelines import StandardScalerND

from .classifier import FNIRSClassifier


def make_dl_pipeline(module, **classifier_params):
    """Create a pipeline with per-chromophore scaling.

    Input must be shaped ``(trials, chromophores, channels, timepoints)``.

    Parameters
    ----------
    module : type of torch.nn.Module
        Neural architecture returning class logits.
    **classifier_params : dict
        Parameters forwarded to FNIRSClassifier.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Scaler and FNIRSClassifier.
    """
    params = {"device": "auto", **classifier_params}
    return make_pipeline(
        StandardScalerND(dim=(0, 2, 3)),
        FNIRSClassifier(module=module, **params),
    )
