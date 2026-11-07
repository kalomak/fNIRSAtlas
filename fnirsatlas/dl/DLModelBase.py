import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin

from fnirsatlas.dl.utils import to_dataloader

logger = logging.getLogger(__name__)


class DLModelBase(nn.Module, BaseEstimator, ClassifierMixin):
    """Base class for deep learning models. SKLearn compatible."""

    def __init__(
        self,
        lr=1e-2,
        lr_step_size=20,
        lr_gamma=0.5,
        label_smoothing=0.0,
        batch_size=32,
        n_epochs=100,
        reset_on_fit=True,
        device="auto",
        random_state=None,
        **kwargs,
    ):
        super().__init__()
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.reset_on_fit = reset_on_fit
        self.device = device
        self.random_state = random_state
        self.val_metrics = defaultdict(list)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.epoch = -1
        self.is_fitted_ = False

    def _resolve_device(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def reset_parameters(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def create_train_dataloader(self, X, y, *, device, **kwargs):
        return to_dataloader(
            X,
            y,
            batch_size=self.batch_size,
            shuffle=True,
            device=device.type,
            **kwargs,
        )

    def create_val_dataloader(self, X, y, *, device, **kwargs):
        return to_dataloader(
            X,
            y,
            batch_size=self.batch_size,
            shuffle=False,
            device=device.type,
            **kwargs,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the model. Resets weights each time."""
        device = self._resolve_device()
        self.to(device)
        logger.info(f"Starting training on device: {device}")
        if self.reset_on_fit:
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self.reset_parameters()
        run_val = X_val is not None and y_val is not None
        if run_val:
            val_dl = self.create_val_dataloader(X_val, y_val, device=device)
            self._validate(val_dl, device)  # pre-validation

        # Train
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )
        epoch_iter = (
            tqdm(range(self.n_epochs), desc="Training epochs", unit="epoch")
            if logger.isEnabledFor(logging.INFO)
            else range(self.n_epochs)
        )
        for epoch in epoch_iter:
            self.train()
            self.epoch = epoch
            train_dl = self.create_train_dataloader(X, y, device=device)
            for batch in train_dl:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

            # Validation
            if run_val:
                self._validate(val_dl, device)
            self._on_epoch_end(optimizer)

        self.is_fitted_ = True

        return self

    def forward(self, x):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)

    def predict_proba(self, X) -> np.ndarray:
        device = self._resolve_device()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(device)
        self.eval()
        with torch.inference_mode():
            outputs = self.forward(X)
            return outputs.detach().cpu().numpy()

    def _validate(self, val_dl, device=None):
        if device is None:
            device = self._resolve_device()
        epoch_val_metrics = defaultdict(list)
        self.eval()
        with torch.inference_mode():
            for batch in val_dl:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                epoch_val_metrics["val_loss"].append(loss.item())
                epoch_val_metrics["y_true"].append(labels.detach().cpu().numpy())
                epoch_val_metrics["y_pred"].append(outputs.detach().cpu().numpy())
        self.on_val_epoch_end(epoch_val_metrics)

    def on_val_epoch_end(self, val_metrics):
        metrics_clean = {}
        for key, values in val_metrics.items():
            if isinstance(values[0], float):
                metrics_clean[key] = np.array(values)
            else:
                metrics_clean[key] = np.concatenate(values)

        y_true = metrics_clean["y_true"]
        y_pred = metrics_clean["y_pred"].argmax(axis=-1)
        val_acc = accuracy_score(y_true, y_pred)
        metrics_clean["val_acc"] = val_acc
        logger.info(f"Epoch {self.epoch}: {val_acc:.3f}")
        for k, v in metrics_clean.items():
            self.val_metrics[k].append(v)

    def _on_epoch_end(self, optimizer):
        pass

    # TODO: replace get_params/set_params with super() calls — BaseEstimator already
    # handles this by inspecting __init__ signature, skipping non-param attributes.
    def get_params(self, deep=True):
        """Get parameters for this estimator. Required for sklearn compatibility."""
        # Get all attributes that are set in __init__ (exclude runtime state and non-param attributes)
        params = {}
        exclude = [
            "val_metrics",
            "scaler",
            "loss_fn",
            "training",
            "epoch",
            "is_fitted_",
            "_parameters",
            "_buffers",
            "_non_persistent_buffers_set",
            "_backward_hooks",
            "_forward_hooks",
            "_forward_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_modules",
        ]
        for key in self.__dict__:
            if not key.startswith("_") and key not in exclude:
                value = getattr(self, key)
                if deep and hasattr(value, "get_params"):
                    params[key] = value
                    nested_params = value.get_params(deep=True)
                    for nested_key, nested_value in nested_params.items():
                        params[f"{key}__{nested_key}"] = nested_value
                else:
                    params[key] = value
        return params

    def set_params(self, **params):
        """Set parameters and rebuild layers via __init__."""
        logger.info(f"Setting parameters: {params}")
        for key, value in params.items():
            setattr(self, key, value)
        self.__init__(**self.get_params())
        return self
