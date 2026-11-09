"""Scikit-learn compatible PyTorch classifier for fNIRS arrays."""

import inspect

import numpy as np
import torch
from mne import BaseEpochs
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fnirsatlas.paradigms import epochs_to_array

_SIGNAL_PARAMETERS = {
    "n_times": lambda X, n_outputs: X.shape[-1],
    "T": lambda X, n_outputs: X.shape[-1],
    "n_channels": lambda X, n_outputs: X.shape[-2],
    "n_chans": lambda X, n_outputs: X.shape[-2],
    "n_chs": lambda X, n_outputs: X.shape[-2],
    "n_chromophores": lambda X, n_outputs: X.shape[-3],
    "n_ch_types": lambda X, n_outputs: X.shape[-3],
    "n_outputs": lambda X, n_outputs: n_outputs,
    "num_classes": lambda X, n_outputs: n_outputs,
}
_PARAMETER_PREFIXES = (
    "module__",
    "criterion__",
    "optimizer__",
    "scheduler__",
)


class FNIRSClassifier(ClassifierMixin, BaseEstimator):
    """Train a PyTorch module through the estimator API expected by MOABB.

    A fresh module is created on every ``fit``. Three-dimensional input gains a
    singleton chromophore axis, so modules always receive ``(trials,
    chromophores, channels, timepoints)`` and must return ``(trials, classes)``
    logits. MNE Epochs input is converted to NumPy and stores ``info_`` and
    ``metadata_`` on the fitted classifier.

    Parameters
    ----------
    module : type of torch.nn.Module
        Uninstantiated module class. Signal dimensions are inferred for
        recognized constructor names such as ``n_channels``, ``n_times``,
        ``n_chromophores``, and ``n_outputs``.
    criterion : type of torch.nn.Module
        Loss class.
    optimizer : type of torch.optim.Optimizer
        Optimizer class.
    scheduler : type or None
        Learning-rate scheduler class.
    lr : float
        Optimizer learning rate.
    max_epochs : int
        Number of training epochs.
    batch_size : int
        Number of trials per batch.
    unit_factor : float
        Scaling applied when converting MNE Epochs.
    device : str or torch.device
        Training device. ``"auto"`` selects CUDA, then MPS, then CPU.
    random_state : int or None
        PyTorch random seed.
    **kwargs : dict
        Component parameters using ``module__``, ``criterion__``,
        ``optimizer__``, or ``scheduler__`` prefixes.
    """

    def __init__(
        self,
        module,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        scheduler=None,
        lr=1e-2,
        max_epochs=100,
        batch_size=32,
        unit_factor=1.0,
        device="auto",
        random_state=None,
        **kwargs,
    ):
        invalid = [key for key in kwargs if not key.startswith(_PARAMETER_PREFIXES)]
        if invalid:
            raise TypeError(f"Unexpected parameter(s): {sorted(invalid)}")
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.unit_factor = unit_factor
        self.device = device
        self.random_state = random_state
        self._kwargs = kwargs

    def _train_epoch(self, loader):
        self.module_.train()
        for inputs, labels in loader:
            self.optimizer_.zero_grad(set_to_none=True)
            logits = self.module_(inputs.to(self.device_))
            self._check_logits(logits)
            loss = self.criterion_(logits, labels.to(self.device_))
            loss.backward()
            self.optimizer_.step()

    def fit(self, X, y=None):
        """Fit a fresh module and return this estimator."""
        self._clear_fitted_state()
        X, epoch_y = self._prepare_X(X, fitting=True)
        if y is None:
            y = epoch_y
        if y is None:
            raise ValueError("y is required when X is not MNE Epochs")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("y must contain at least two classes")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.device_ = self._resolve_device()
        self.module_ = self._build_module(X).to(self.device_)
        self.criterion_ = self.criterion(**self._component_params("criterion")).to(
            self.device_
        )
        optimizer_params = self._component_params("optimizer")
        optimizer_params.setdefault("lr", self.lr)
        self.optimizer_ = self.optimizer(self.module_.parameters(), **optimizer_params)
        scheduler_params = self._component_params("scheduler")
        if self.scheduler is None and scheduler_params:
            raise ValueError("scheduler__ parameters require a scheduler")
        self.scheduler_ = (
            None
            if self.scheduler is None
            else self.scheduler(self.optimizer_, **scheduler_params)
        )
        loader = self._loader(X, encoded_y, shuffle=True)
        for _ in range(self.max_epochs):
            self._train_epoch(loader)
            if self.scheduler_ is not None:
                self.scheduler_.step()
        self.input_shape_ = X.shape[1:]
        self.n_features_in_ = int(np.prod(X.shape[1:]))
        return self

    def _prepare_X(self, X, *, fitting=False):
        epoch_y = None
        if isinstance(X, BaseEpochs):
            X, epoch_y, metadata, info = epochs_to_array(X, self.unit_factor)
            if fitting:
                self.metadata_ = metadata
                self.info_ = info
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (3, 4):
            raise ValueError(f"X must be 3D or 4D, got shape {X.shape}")
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        if not len(X):
            raise ValueError("X must contain at least one sample")
        if not fitting and X.shape[1:] != self.input_shape_:
            raise ValueError(
                f"Expected input shape {self.input_shape_}, got {X.shape[1:]}"
            )
        return np.ascontiguousarray(X), epoch_y

    def _loader(self, X, y=None, *, shuffle=False):
        tensors = [torch.from_numpy(X)]
        if y is not None:
            tensors.append(torch.as_tensor(y, dtype=torch.long))
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        return DataLoader(
            TensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _check_logits(self, logits):
        expected = len(self.classes_)
        if logits.ndim != 2 or logits.shape[1] != expected:
            raise ValueError(f"Module must return (samples, {expected}) logits")

    def predict_proba(self, X):
        """Return class probabilities for each sample."""
        check_is_fitted(self, "module_")
        X, _ = self._prepare_X(X)
        probabilities = []
        self.module_.eval()
        with torch.inference_mode():
            for (inputs,) in self._loader(X):
                logits = self.module_(inputs.to(self.device_))
                self._check_logits(logits)
                probabilities.append(logits.softmax(dim=1).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, X):
        """Return predicted labels in the original label representation."""
        check_is_fitted(self, "module_")
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def get_params(self, deep=True):
        """Return estimator and prefixed component parameters."""
        params = super().get_params(deep=False)
        params.update(self._kwargs)
        return params

    def set_params(self, **params):
        """Set estimator or prefixed component parameters."""
        valid = self.get_params(deep=False)
        invalid = [
            key
            for key in params
            if key not in valid and not key.startswith(_PARAMETER_PREFIXES)
        ]
        if invalid:
            raise ValueError(f"Invalid parameter(s): {sorted(invalid)}")
        for key, value in params.items():
            if key.startswith(_PARAMETER_PREFIXES):
                self._kwargs[key] = value
            else:
                setattr(self, key, value)
        self._clear_fitted_state()
        return self

    def __sklearn_tags__(self):
        """Declare the multidimensional input expected by fNIRS models."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        return tags

    def _clear_fitted_state(self):
        for name in (
            "classes_",
            "criterion_",
            "device_",
            "input_shape_",
            "info_",
            "metadata_",
            "module_",
            "n_features_in_",
            "optimizer_",
            "scheduler_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _component_params(self, component):
        prefix = f"{component}__"
        return {
            key.removeprefix(prefix): value
            for key, value in self._kwargs.items()
            if key.startswith(prefix)
        }

    def _build_module(self, X):
        if not inspect.isclass(self.module) or not issubclass(self.module, nn.Module):
            raise TypeError("module must be a torch.nn.Module class")
        parameters = self._component_params("module")
        signature = inspect.signature(self.module)
        for name, resolver in _SIGNAL_PARAMETERS.items():
            if name not in signature.parameters:
                continue
            inferred = resolver(X, len(self.classes_))
            if name in parameters and parameters[name] != inferred:
                raise ValueError(f"module__{name} must equal inferred value {inferred}")
            parameters[name] = inferred
        return self.module(**parameters)

    def _resolve_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
