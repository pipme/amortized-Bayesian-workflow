"""LC2ST implementation using skorch for optional GPU-accelerated MLP. This is not used in the manuscript since we empirically found LC2ST(-NF) is sensitive to various factors. We also compared with the original LC2ST implementation to check the correctness of implementation. We keep this implementation for future reference and potential use in other tasks."""
# ruff: noqa: UP035, UP006, UP007

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import skorch
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from torch import Tensor, ones
from tqdm.auto import tqdm

# --- Helper Functions from original lc2st.py (potentially move to utils) ---


def handle_invalid_x(
    x: Tensor, exclude_invalid_x: bool = True
) -> Tuple[Tensor, int, int]:
    """Return Tensor mask that is True where simulations `x` are valid."""
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1)
    x_is_nan = torch.isnan(x).any(dim=1)
    x_is_inf = torch.isinf(x).any(dim=1)
    num_nans = int(x_is_nan.sum().item())
    num_infs = int(x_is_inf.sum().item())
    if exclude_invalid_x:
        is_valid_x = ~x_is_nan & ~x_is_inf
    else:
        is_valid_x = ones(batch_size, dtype=torch.bool, device=x.device)
    if is_valid_x.sum() == 0 and exclude_invalid_x:
        raise ValueError(
            "No valid data entries left after excluding NaNs and Infs."
        )
    return is_valid_x, num_nans, num_infs


def remove_nans_and_infs_in_x(
    thetas: Tensor, xs: Tensor
) -> Tuple[Tensor, Tensor]:
    """Remove NaNs and Infs entries in x from both the theta and x."""
    is_valid_x, num_nans, num_infs = handle_invalid_x(
        xs, exclude_invalid_x=True
    )
    num_samples_orig = len(xs)
    num_samples_valid = int(is_valid_x.sum().item())

    if num_nans > 0 or num_infs > 0:
        warnings.warn(
            f"Found {num_nans} NaNs and {num_infs} Infs in the data (xs). "
            f"These simulation outputs will be ignored. "
            f"Keeping {num_samples_valid} / {num_samples_orig} samples.",
            stacklevel=2,
        )

    return thetas[is_valid_x], xs[is_valid_x]


def permute_data(
    joint_p: Tensor, joint_q: Tensor, seed: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Permutes the concatenated data [P,Q] to create null samples."""
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    assert joint_p.shape[0] == joint_q.shape[0]
    sample_size = joint_p.shape[0]
    X = torch.cat([joint_p, joint_q], dim=0)
    x_perm = X[torch.randperm(sample_size * 2, device=X.device)]

    if seed is not None:
        torch.set_rng_state(rng_state)

    return x_perm[:sample_size], x_perm[sample_size:]


# --- PyTorch MLP Definition ---


class TorchMLP(nn.Module):
    """A configurable PyTorch MLP for binary classification with multiple hidden layers."""

    def __init__(
        self,
        num_inputs: int,
        num_hidden: int = 50,
        num_outputs: int = 2,
        num_layers: int = 1,  # Number of hidden layers
        dropout_rate: float = 0.0,  # Optional dropout between layers
    ):
        super().__init__()

        # Input layer
        layers = [
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(num_hidden, num_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )

        # Output layer
        layers.append(nn.Linear(num_hidden, num_outputs))

        # Combine all layers into a Sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)  # Return logits for CrossEntropyLoss


# --- Ensemble Classifier Wrapper ---


class EnsembleClassifier(BaseEstimator):
    """Wraps multiple classifiers (sklearn or skorch) for ensembling."""

    # Needs BaseEstimator for clone to work easily
    def __init__(
        self,
        clf_template: Union[BaseEstimator, NeuralNetClassifier],
        num_ensemble: int = 1,
        verbosity: int = 1,
    ):
        self.clf_template = clf_template
        self.num_ensemble = num_ensemble
        self.verbosity = verbosity
        self.trained_clfs_: List[
            Union[BaseEstimator, NeuralNetClassifier]
        ] = []
        # Store parameters for get_params/set_params used by clone
        self._init_kwargs = {
            "clf_template": clf_template,
            "num_ensemble": num_ensemble,
            "verbosity": verbosity,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """Fits the ensemble of classifiers."""
        self.trained_clfs_ = []
        base_seed = None
        is_sklearn = isinstance(
            self.clf_template, BaseEstimator
        ) and not isinstance(self.clf_template, NeuralNetClassifier)

        # Try to get base seed if sklearn classifier has random_state
        if is_sklearn and hasattr(self.clf_template, "random_state"):
            base_seed = self.clf_template.random_state  # type: ignore

        for n in tqdm(
            range(self.num_ensemble),
            desc="Ensemble training",
            disable=self.verbosity < 1 or self.num_ensemble <= 1,
        ):
            clf_n = clone(self.clf_template)

            # Handle seeding consistently
            current_seed = base_seed + n if base_seed is not None else None

            if is_sklearn:
                if hasattr(clf_n, "random_state"):
                    clf_n.random_state = current_seed  # type: ignore
            elif isinstance(clf_n, NeuralNetClassifier):
                # Skorch/PyTorch seeding is more complex.
                # For basic reproducibility, set torch seed before fit.
                # Skorch itself doesn't have a simple top-level random_state for PyTorch modules.
                if current_seed is not None:
                    torch.manual_seed(current_seed)
                # Could also pass random_state to skorch callbacks if needed, e.g., for data splitting

            clf_n.fit(X, y)
            self.trained_clfs_.append(clf_n)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Averages predicted probabilities from the ensemble."""
        if not self.trained_clfs_:
            raise RuntimeError("Classifier has not been fitted.")

        # Ensure X is float32 for skorch/torch models
        if not isinstance(X, np.ndarray) or X.dtype != np.float32:
            X = np.asarray(X, dtype=np.float32)

        all_probas = [clf.predict_proba(X) for clf in self.trained_clfs_]
        return np.mean(all_probas, axis=0)

    def get_params(self, deep=True):
        """Gets parameters for this estimator."""
        # Return the stored init kwargs so clone works
        return self._init_kwargs

    def set_params(self, **params):
        """Sets the parameters of this estimator."""
        # Update stored kwargs and attributes
        self._init_kwargs.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self


# --- Main LC2ST Class ---


class LC2ST:
    def __init__(
        self,
        thetas: Tensor,
        xs: Tensor,
        posterior_samples: Tensor,
        seed: int = 1,
        num_folds: int = 1,
        num_ensemble: int = 1,
        classifier: Union[
            str,
            Type[BaseEstimator],
            Type[NeuralNetClassifier],
            BaseEstimator,
            NeuralNetClassifier,
        ] = "mlp",
        z_score: bool = False,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        num_trials_null: int = 100,
        permutation: bool = True,
        device: Optional[str] = None,  # Add device parameter
    ) -> None:
        """
        L-C2ST: Local Classifier Two-Sample Test (with skorch/GPU support for MLP)
        -------------------------------------------------------------------------
        Implementation based on the official code from [1] and the exisiting C2ST
        metric [2], using scikit-learn classifiers or skorch-wrapped PyTorch MLPs.

        Args:
            thetas: Samples from the prior distribution (or reference distribution q),
                    shape (N, D_theta).
            xs: Corresponding observations/simulations for thetas, shape (N, D_x).
            posterior_samples: Samples from the estimated posterior (or target distribution p),
                               shape (N, D_theta). Assumed to correspond to the same xs.
            seed: Seed for reproducibility (classifier init, kfold, permutation), defaults to 1.
            num_folds: Number of folds for cross-validation, defaults to 1.
            num_ensemble: Number of classifiers in the ensemble, defaults to 1.
            classifier: Specifies the classifier. Can be:
                        - "mlp": Uses skorch `NeuralNetClassifier` with `TorchMLP` (GPU if available).
                        - "random_forest": Uses sklearn `RandomForestClassifier`.
                        - A scikit-learn `BaseEstimator` class or instance.
                        - A skorch `NeuralNetClassifier` class or instance.
                        Defaults to "mlp".
            z_score: Whether to z-score thetas and xs based on posterior_samples and xs,
                     defaults to False.
            classifier_kwargs: Dictionary of keyword arguments passed to the classifier
                               constructor (or skorch `NeuralNetClassifier`), defaults to None.
            num_trials_null: Number of trials to estimate the null distribution, defaults to 100.
            permutation: Whether to use permutation testing for the null distribution,
                         defaults to True. Ignored if `LC2ST_NF` is used.
            device: PyTorch device ('cuda', 'cpu', etc.). If None, defaults to 'cuda' if
                    available, else 'cpu'. Only relevant for 'mlp' classifier.

        References:
        [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
        [2] : https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py
        """
        # Determine device for PyTorch operations
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Convert numpy arrays to torch tensors if needed
        if isinstance(thetas, np.ndarray):
            thetas = torch.from_numpy(thetas).float()
        if isinstance(xs, np.ndarray):
            xs = torch.from_numpy(xs).float()
        if isinstance(posterior_samples, np.ndarray):
            posterior_samples = torch.from_numpy(posterior_samples).float()
        thetas = thetas.to(self.device)
        xs = xs.to(self.device)
        posterior_samples = posterior_samples.to(self.device)

        thetas, xs = remove_nans_and_infs_in_x(thetas, xs)
        # Also check posterior samples (though less likely to have NaNs from simulator)
        # Assuming posterior_samples don't have NaNs/Infs from the inference process itself
        if (
            torch.isnan(posterior_samples).any()
            or torch.isinf(posterior_samples).any()
        ):
            warnings.warn(
                "NaNs or Infs found in posterior_samples.", stacklevel=2
            )

        assert thetas.shape[0] == xs.shape[0] == posterior_samples.shape[0], (
            "Number of samples must match for thetas, xs, and posterior_samples."
        )
        assert thetas.ndim == 2, "thetas must be 2D (samples, features)"
        assert xs.ndim == 2, "xs must be 2D (samples, features)"
        assert posterior_samples.ndim == 2, (
            "posterior_samples must be 2D (samples, features)"
        )

        self.seed = seed
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble
        self.z_score = z_score
        self.num_trials_null = num_trials_null
        self.permutation = permutation  # Used by train_under_null_hypothesis

        # Store data (use posterior samples as p, prior samples as q)
        self.theta_p = posterior_samples  # Samples from target p(theta|x)
        self.theta_q = (
            thetas  # Samples from reference q(theta|x) (e.g., prior)
        )
        self.x_p = xs  # Observations x corresponding to theta_p
        self.x_q = xs  # Observations x corresponding to theta_q

        # Calculate z-score stats based on the 'p' distribution (posterior samples)
        # These stats will be used to normalize all data (p, q, and test data)
        self.theta_mean = torch.mean(self.theta_p, dim=0, keepdim=True)
        self.theta_std = torch.std(self.theta_p, dim=0, keepdim=True)
        self.x_mean = torch.mean(self.x_p, dim=0, keepdim=True)
        self.x_std = torch.std(self.x_p, dim=0, keepdim=True)
        # Avoid division by zero if std is 0
        self.theta_std[self.theta_std == 0] = 1.0
        self.x_std[self.x_std == 0] = 1.0

        # --- Initialize Classifier Template ---
        classifier_kwargs = classifier_kwargs or {}
        clf_template: Union[BaseEstimator, NeuralNetClassifier]

        if isinstance(classifier, str):
            clf_str = classifier.lower()
            if clf_str == "mlp":
                ndim_theta = self.theta_p.shape[-1]
                ndim_x = self.x_p.shape[-1]
                input_dim = ndim_theta + ndim_x
                print("Input dim: ", input_dim)
                default_mlp_kwargs = {
                    "module": TorchMLP,
                    "module__num_inputs": input_dim,
                    "module__num_hidden": 10 * input_dim,  # Default sizing
                    "criterion": nn.CrossEntropyLoss,  # Expects logits
                    "optimizer": torch.optim.Adam,
                    "lr": 1e-3,
                    "max_epochs": 100,  # Adjusted default
                    "batch_size": 128,
                    "device": self.device,
                    "callbacks": [
                        (
                            "early_stopping",
                            EarlyStopping(patience=10, monitor="valid_loss"),
                        ),  # Monitor validation loss
                        # Add accuracy scoring
                        (
                            "accuracy",
                            EpochScoring(
                                "accuracy",
                                on_train=True,
                            ),
                        ),
                        # Add TrainEndCheckpoint? Checkpoint?
                    ],
                    "train_split": skorch.dataset.ValidSplit(
                        cv=5, stratified=True
                    )
                    if self.num_folds == 1
                    else None,  # Use internal validation split if not doing CV externally
                    "verbose": 1,  # Skorch verbosity (0 = silent)
                    "iterator_train__shuffle": True,
                }
                merged_kwargs = {**default_mlp_kwargs, **classifier_kwargs}
                clf_template = NeuralNetClassifier(**merged_kwargs)

            elif clf_str == "random_forest":
                default_rf_kwargs = {"random_state": self.seed}
                merged_kwargs = {**default_rf_kwargs, **classifier_kwargs}
                clf_template = RandomForestClassifier(**merged_kwargs)
            else:
                raise ValueError(f"Unknown classifier string: '{classifier}'")

        elif isinstance(
            classifier, (type(BaseEstimator), type(NeuralNetClassifier))
        ):
            # If a class is passed
            if issubclass(classifier, BaseEstimator) and not issubclass(
                classifier, NeuralNetClassifier
            ):
                # Standard sklearn estimator class
                default_sk_kwargs = (
                    {"random_state": self.seed}
                    if hasattr(classifier(), "random_state")
                    else {}
                )
                merged_kwargs = {**default_sk_kwargs, **classifier_kwargs}
                clf_template = classifier(**merged_kwargs)
            elif issubclass(classifier, NeuralNetClassifier):
                # Skorch class - user must provide necessary module args etc. in kwargs
                clf_template = classifier(**classifier_kwargs)
            else:
                raise TypeError(
                    f"Unsupported classifier class type: {classifier}"
                )

        elif isinstance(classifier, (BaseEstimator, NeuralNetClassifier)):
            # If an instance is passed
            clf_template = classifier  # Use the provided instance as template
            # Optionally update random_state if applicable and not already set in instance
            if isinstance(clf_template, BaseEstimator) and not isinstance(
                clf_template, NeuralNetClassifier
            ):
                if (
                    hasattr(clf_template, "random_state")
                    and getattr(clf_template, "random_state", None) is None
                ):
                    try:
                        clf_template.set_params(random_state=self.seed)
                    except (
                        ValueError
                    ):  # Some estimators might not accept random_state
                        pass
        else:
            raise TypeError(f"Invalid classifier type: {type(classifier)}")

        # --- Initialize Ensemble Template (if needed) ---
        if self.num_ensemble > 1:
            # Wrap the single classifier template in an EnsembleClassifier
            self.clf_template = EnsembleClassifier(
                clf_template=clf_template,
                num_ensemble=self.num_ensemble,
                verbosity=1,  # Default verbosity for ensemble
            )
        else:
            self.clf_template = (
                clf_template  # Use the single classifier directly
            )

        # Placeholders for trained classifiers
        self.trained_clfs_: Optional[
            List[Union[BaseEstimator, NeuralNetClassifier]]
        ] = None
        self.trained_clfs_null_: Optional[
            Dict[int, List[Union[BaseEstimator, NeuralNetClassifier]]]
        ] = None

        # Placeholder for null distribution (used by LC2ST_NF)
        self.null_distribution: Optional[torch.distributions.Distribution] = (
            None
        )
        self.num_eval: Optional[int] = (
            None  # Number of evaluations (for LC2ST_NF)
        )
        # Add placeholder for base samples used in NF evaluation
        self.theta_o_base: Optional[Tensor] = None

    def _prepare_data_for_clf(
        self, theta_p: Tensor, theta_q: Tensor, x_p: Tensor, x_q: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies z-scoring (if enabled) and concatenates/converts data for classifier input."""

        if self.z_score:
            # Normalize using stats calculated in __init__
            theta_p = (theta_p - self.theta_mean) / self.theta_std
            theta_q = (theta_q - self.theta_mean) / self.theta_std
            x_p = (x_p - self.x_mean) / self.x_std
            x_q = (x_q - self.x_mean) / self.x_std

        # Concatenate features: (theta, x)
        joint_p = torch.cat([theta_p, x_p], dim=1)
        joint_q = torch.cat([theta_q, x_q], dim=1)

        # Combine P and Q samples
        data = torch.cat([joint_p, joint_q], dim=0)
        # Create labels: 0 for P, 1 for Q
        target = torch.cat(
            [
                torch.zeros(joint_p.shape[0], device=joint_p.device),
                torch.ones(joint_q.shape[0], device=joint_q.device),
            ],
            dim=0,
        )

        # Convert to numpy arrays required by sklearn/skorch interface
        # Use float32 for data (common for NNs) and int64 for targets (for CrossEntropyLoss)
        data_np = data.cpu().numpy().astype(np.float32)
        target_np = target.cpu().numpy().astype(np.int64)

        return data_np, target_np

    def _train(
        self,
        theta_p: Tensor,
        theta_q: Tensor,
        x_p: Tensor,
        x_q: Tensor,
        seed_offset: int = 0,  # Offset for fold/trial specific seeding
        verbosity: int = 0,
    ) -> List[Union[BaseEstimator, NeuralNetClassifier]]:
        """Trains classifier(s) on the provided data splits (P vs Q)."""

        data_full, target_full = self._prepare_data_for_clf(
            theta_p, theta_q, x_p, x_q
        )
        trained_clfs = []

        if self.num_folds > 1:
            kf = KFold(
                n_splits=self.num_folds,
                shuffle=True,
                random_state=self.seed + seed_offset,
            )
            cv_splits = kf.split(data_full)

            for fold_idx, (train_idx, _) in enumerate(
                tqdm(
                    cv_splits,
                    total=self.num_folds,
                    desc=f"CV Folds (Seed Offset {seed_offset})",
                    disable=verbosity < 1 or self.num_folds <= 1,
                    leave=False,
                )
            ):
                data_train, target_train = (
                    data_full[train_idx],
                    target_full[train_idx],
                )

                # Clone the template (which might be an EnsembleClassifier) for this fold
                clf_fold = clone(self.clf_template)

                # Adjust seed for the fold if the classifier supports it
                fold_seed = self.seed + seed_offset + fold_idx
                if isinstance(clf_fold, EnsembleClassifier):
                    # Ensemble handles internal seeding based on its template's state
                    pass  # Seed handled within EnsembleClassifier.fit
                elif isinstance(clf_fold, BaseEstimator) and not isinstance(
                    clf_fold, NeuralNetClassifier
                ):
                    if hasattr(clf_fold, "random_state"):
                        clf_fold.set_params(random_state=fold_seed)
                elif isinstance(clf_fold, NeuralNetClassifier):
                    # Set torch seed for this fold's training
                    torch.manual_seed(fold_seed)

                clf_fold.fit(data_train, target_train)
                trained_clfs.append(clf_fold)
        else:
            # No cross-validation, train on full data
            clf_single = clone(self.clf_template)
            single_seed = self.seed + seed_offset

            # Adjust seed
            if isinstance(clf_single, EnsembleClassifier):
                pass  # Seed handled within EnsembleClassifier.fit
            elif isinstance(clf_single, BaseEstimator) and not isinstance(
                clf_single, NeuralNetClassifier
            ):
                if hasattr(clf_single, "random_state"):
                    clf_single.set_params(random_state=single_seed)
            elif isinstance(clf_single, NeuralNetClassifier):
                torch.manual_seed(single_seed)

            clf_single.fit(data_full, target_full)
            trained_clfs = [clf_single]

        return trained_clfs

    def _prepare_eval_data(
        self, theta_o: Tensor, x_o: Tensor
    ) -> Tuple[np.ndarray, int, int]:
        """Prepares data for evaluation (z-score, repeat x_o, concatenate, convert).

        Handles both single observation and batched observations.

        Args:
            theta_o: Samples from the posterior (or target distribution p),
                     shape (M, D_theta) or (B, M, D_theta).
            x_o: The observation(s), shape (D_x), (1, D_x), or (B, D_x).

        Returns:
            Tuple of (joint_eval_np, B, M):
            - joint_eval_np: Concatenated and processed data for classifier,
                             shape (B * M, D_theta + D_x).
            - B: Batch size.
            - M: Number of samples per observation.
        """
        # --- Input Shape Handling ---
        if x_o.ndim == 1:
            x_o = x_o.unsqueeze(0)  # Shape (1, D_x)
        B = x_o.shape[0]  # Batch size

        if theta_o.ndim == 2:
            # Case 1: Single set of theta samples (M, D_theta)
            # This might be self.theta_o_base from LC2ST_NF, needs repeating for batch B
            M = theta_o.shape[0]
            if B > 1:
                # Repeat theta_o B times if x_o is batched
                theta_o = theta_o.unsqueeze(0).expand(
                    B, M, -1
                )  # Shape (B, M, D_theta)
            else:
                # Single observation, single theta set
                theta_o = theta_o.unsqueeze(0)  # Shape (1, M, D_theta)
        elif theta_o.ndim == 3:
            # Case 2: Batched theta samples (B, M, D_theta)
            M = theta_o.shape[1]
            if B != theta_o.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: x_o has {B}, theta_o has {theta_o.shape[0]}"
                )
        else:
            raise ValueError(f"Invalid theta_o dimensions: {theta_o.ndim}")

        # --- Repeat x_o ---
        # x_o shape is (B, D_x), theta_o shape is (B, M, D_theta)
        # Repeat x_o M times for each item in the batch
        x_o_rep = x_o.unsqueeze(1).expand(-1, M, -1)  # Shape (B, M, D_x)

        # --- Flatten ---
        theta_o_flat = theta_o.reshape(B * M, -1)  # Shape (B*M, D_theta)
        x_o_flat = x_o_rep.reshape(B * M, -1)  # Shape (B*M, D_x)

        # --- Z-scoring ---
        if self.z_score:
            # Ensure stats are on the same device
            theta_mean_dev = self.theta_mean.to(theta_o_flat.device)
            theta_std_dev = self.theta_std.to(theta_o_flat.device)
            x_mean_dev = self.x_mean.to(x_o_flat.device)
            x_std_dev = self.x_std.to(x_o_flat.device)

            theta_o_flat = (theta_o_flat - theta_mean_dev) / theta_std_dev
            x_o_flat = (x_o_flat - x_mean_dev) / x_std_dev

        # --- Concatenate and Convert ---
        joint_eval = torch.cat(
            [theta_o_flat, x_o_flat], dim=1
        )  # Shape (B*M, D_theta + D_x)
        joint_eval_np = joint_eval.cpu().numpy().astype(np.float32)

        return joint_eval_np, B, M

    def _eval_single_clf(
        self,
        clf: Union[BaseEstimator, NeuralNetClassifier],
        joint_eval_np: np.ndarray,
        B: int,
        M: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates a single trained classifier on potentially batched data.

        Args:
            clf: The trained classifier.
            joint_eval_np: Prepared evaluation data, shape (B * M, D_features).
            B: Batch size.
            M: Samples per observation.

        Returns:
            Tuple of (probabilities, scores):
            - probabilities: Predicted probabilities for class 0 (P), shape (B, M).
            - scores: L-C2ST scores for each observation in the batch, shape (B,).
        """
        # predict_proba returns [[prob_class_0, prob_class_1], ...]
        # Shape: (B * M, 2)
        proba_flat = clf.predict_proba(joint_eval_np)
        # Probability of belonging to class 0 (P)
        # Shape: (B * M,)
        proba_p_flat = proba_flat[:, 0]
        # Reshape to (B, M)
        proba_p = proba_p_flat.reshape(B, M)

        # L-C2ST score: MSE between predicted prob(class 0) and 0.5, averaged over M samples
        # Calculate mean score per observation (axis=1)
        # Shape: (B,)
        scores = np.mean((proba_p - 0.5) ** 2, axis=1)

        return proba_p, scores

    def get_scores(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        trained_clfs: List[Union[BaseEstimator, NeuralNetClassifier]],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes L-C2ST scores for given batch of data and trained classifiers.

        Args:
            theta_o: Samples from the posterior (or target distribution p) conditioned
                     on `x_o`, shape (M, D_theta) or (B, M, D_theta).
            x_o: The observation(s), shape (D_x), (1, D_x), or (B, D_x).
            trained_clfs: List of trained classifiers (one per fold).
            return_probs: Whether to return predicted probabilities (class 0), defaults to False.

        Returns:
            If return_probs is False: L-C2ST scores, shape (B, num_folds).
            If return_probs is True: Tuple of (probabilities, scores),
                                     shapes ((B, num_folds, M), (B, num_folds)).
        """
        # Ensure data is on the correct device before preparation
        device = self.device
        theta_o = theta_o.to(device)
        x_o = x_o.to(device)

        joint_eval_np, B, M = self._prepare_eval_data(theta_o, x_o)

        all_probs_list = []  # List of arrays, each shape (B, M)
        all_scores_list = []  # List of arrays, each shape (B,)

        for clf in trained_clfs:
            proba_p, scores = self._eval_single_clf(clf, joint_eval_np, B, M)
            all_probs_list.append(proba_p)
            all_scores_list.append(scores)

        # Stack scores across folds: list of (B,) arrays -> (B, num_folds)
        scores_arr = np.stack(all_scores_list, axis=1)

        if return_probs:
            # Stack probabilities across folds: list of (B, M) arrays -> (B, num_folds, M)
            probs_arr = np.stack(all_probs_list, axis=1)
            return probs_arr, scores_arr
        else:
            return scores_arr  # Shape (B, num_folds)

    def train_on_observed_data(self, verbosity: int = 1) -> "LC2ST":
        """Trains the classifier(s) on the observed data (p vs q).

        Uses `self.theta_p`, `self.theta_q`, `self.x_p`, `self.x_q`.
        Stores the trained classifier(s) in `self.trained_clfs_`.

        Args:
            verbosity: Verbosity level for progress bars, defaults to 1.

        Returns:
            self
        """
        if self.trained_clfs_ is not None:
            warnings.warn(
                "Classifiers already trained on observed data. Retraining.",
                stacklevel=2,
            )

        print("Training classifier(s) on observed data (p vs q)...")
        self.trained_clfs_ = self._train(
            self.theta_p,
            self.theta_q,
            self.x_p,
            self.x_q,
            seed_offset=0,
            verbosity=verbosity,
        )
        return self

    def get_statistic_on_observed_data(
        self,
        theta_o: Tensor,
        x_o: Tensor,
    ) -> np.ndarray:
        """Computes the mean L-C2ST statistic over folds for the observed data batch.

        Args:
            theta_o: Samples from the posterior conditioned on `x_o`, shape (B, M, D_theta).
            x_o: The observation batch, shape (B, D_x).

        Returns:
            Mean L-C2ST statistics T_data for each observation, shape (B,).
        """
        if self.trained_clfs_ is None:
            raise RuntimeError(
                "No trained classifiers found for observed data. Run `train_on_observed_data` first."
            )
        # get_scores returns shape (B, num_folds)
        scores = self.get_scores(
            theta_o=theta_o,
            x_o=x_o,
            trained_clfs=self.trained_clfs_,
            return_probs=False,
        )
        # Mean over folds (axis=1) -> shape (B,)
        return scores.mean(axis=1)

    def train_under_null_hypothesis(self, verbosity: int = 1) -> "LC2ST":
        """Trains classifiers under the null hypothesis H0: p=q.

        Uses permutation testing (if `self.permutation=True`) or samples from
        `self.null_distribution` (if available, e.g., in `LC2ST_NF`).
        Stores trained classifiers for each trial in `self.trained_clfs_null_`.

        Args:
            verbosity: Verbosity level for progress bars, defaults to 1.

        Returns:
            self
        """
        if self.trained_clfs_null_ is not None:
            warnings.warn(
                "Classifiers already trained under null. Retraining.",
                stacklevel=2,
            )

        self.trained_clfs_null_ = {}
        desc = f"Training under H0 (permutation={self.permutation})"
        if not self.permutation and self.null_distribution is None:
            raise ValueError(
                "Permutation is False, but no null_distribution provided."
            )

        print(f"Training {self.num_trials_null} classifier(s) under H0...")
        for t in tqdm(
            range(self.num_trials_null),
            desc=desc,
            disable=verbosity < 1,
        ):
            trial_seed_offset = (t + 1) * (
                self.num_folds + 1
            )  # Ensure different seeds per trial

            if self.permutation:
                # Create permuted data for this trial
                joint_p = torch.cat([self.theta_p, self.x_p], dim=1)
                joint_q = torch.cat([self.theta_q, self.x_q], dim=1)
                joint_p_perm, joint_q_perm = permute_data(
                    joint_p, joint_q, seed=self.seed + trial_seed_offset
                )
                # Extract permuted thetas and xs
                theta_p_t = joint_p_perm[:, : self.theta_p.shape[1]]
                x_p_t = joint_p_perm[:, self.theta_p.shape[1] :]
                theta_q_t = joint_q_perm[:, : self.theta_q.shape[1]]
                x_q_t = joint_q_perm[:, self.theta_q.shape[1] :]
            else:
                # Sample from the known null distribution (e.g., base distribution for NF)
                num_samples = self.theta_p.shape[
                    0
                ]  # Use the same number of samples
                # Ensure null_distribution is defined (checked above)
                theta_p_t = self.null_distribution.sample((num_samples,)).to(
                    self.device
                )  # type: ignore
                theta_q_t = self.null_distribution.sample((num_samples,)).to(
                    self.device
                )  # type: ignore
                # Keep the original xs, as H0 is about theta distribution conditional on x
                x_p_t, x_q_t = (
                    self.x_p.to(self.device),
                    self.x_q.to(self.device),
                )

            # Train classifier(s) for this null trial
            clf_t = self._train(
                theta_p_t,
                theta_q_t,
                x_p_t,
                x_q_t,
                seed_offset=trial_seed_offset,
                verbosity=0,  # Less verbose inside loop
            )
            self.trained_clfs_null_[t] = clf_t

        return self

    def get_statistics_under_null_hypothesis(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        return_probs: bool = False,
        verbosity: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes L-C2ST statistics for each null trial for a batch of observations.

        Args:
            theta_o: Samples from the posterior conditioned on `x_o`, shape (B, M, D_theta).
                     Used only if `self.permutation=True`. If `self.permutation=False`,
                     `self.theta_o_base` (shape (M, D_theta)) is used internally.
            x_o: The observation batch, shape (B, D_x).
            return_probs: Whether to return probabilities for each trial, defaults to False.
            verbosity: Verbosity level for progress bars, defaults to 0.

        Returns:
            If return_probs is False: Array of mean scores T_null for each trial,
                                     shape (B, num_trials_null).
            If return_probs is True: Tuple of (probabilities, scores), where probs have
                                    shape (B, num_trials_null, num_folds, M) and scores
                                    have shape (B, num_trials_null).
        """
        if self.trained_clfs_null_ is None:
            raise RuntimeError(
                "Classifiers under H0 not trained. Run `train_under_null_hypothesis` first."
            )
        if len(self.trained_clfs_null_) != self.num_trials_null:
            warnings.warn(
                f"Found {len(self.trained_clfs_null_)} trained null classifiers, expected {self.num_trials_null}.",
                stacklevel=2,
            )

        all_trial_probs_list = []  # List of arrays, shape (B, num_folds, M)
        all_trial_stats_list = []  # List of arrays, shape (B,)
        desc = f"Evaluating under H0 (permutation={self.permutation})"

        x_o = x_o.to(self.device)
        if x_o.ndim == 1:
            x_o = x_o.unsqueeze(0)
        B = x_o.shape[0]

        for t in tqdm(
            range(self.num_trials_null),
            desc=desc,
            disable=verbosity < 1,
        ):
            if t not in self.trained_clfs_null_:
                warnings.warn(
                    f"Missing trained classifier for null trial {t}. Skipping.",
                    stacklevel=2,
                )
                # Append NaNs or handle appropriately? For now, skip.
                # Need consistent shapes if returning arrays. Let's append NaNs of correct shape.

                num_folds = len(
                    next(iter(self.trained_clfs_null_.values()))
                )  # Infer num_folds
                M = (
                    theta_o.shape[1]
                    if self.permutation
                    else (
                        self.theta_o_base.shape[0]
                        if self.theta_o_base is not None
                        else 0
                    )
                )

                if return_probs:
                    all_trial_probs_list.append(
                        np.full((B, num_folds, M), np.nan)
                    )
                all_trial_stats_list.append(np.full((B,), np.nan))
                continue

            clfs_t = self.trained_clfs_null_[t]

            # Determine which theta samples to use for evaluation under H0
            if self.permutation:
                # theta_o has shape (B, M, D_theta)
                theta_eval_t = theta_o
                M = theta_eval_t.shape[1]  # Number of samples per observation
            else:
                # Use pre-sampled base distribution samples (self.theta_o_base)
                # Shape (M, D_theta). _prepare_eval_data will handle expansion.
                if self.use_presampled_base_for_eval:
                    if self.theta_o_base is None:
                        raise RuntimeError(
                            "`self.theta_o_base` samples for evaluation under known null not found. This should be set by LC2ST_NF."
                        )
                    theta_eval_t = self.theta_o_base
                else:
                    M = self.num_eval  # Number of samples per observation
                    theta_eval_t = self.null_distribution.sample((B, M)).to(
                        self.device
                    )

            # Get scores for this trial. Returns shape (B, num_folds)
            # If return_probs=True, probs_t shape is (B, num_folds, M)
            probs_t, scores_t = self.get_scores(
                theta_o=theta_eval_t,  # Shape (B, M, D_theta) or (M, D_theta)
                x_o=x_o,  # Shape (B, D_x)
                trained_clfs=clfs_t,
                return_probs=True,  # Always get probs here, decide return later
            )
            if return_probs:
                all_trial_probs_list.append(probs_t)  # Shape (B, num_folds, M)

            # Mean score over folds for this trial -> shape (B,)
            all_trial_stats_list.append(scores_t.mean(axis=1))

        # Stack stats across trials: list of (B,) arrays -> (B, num_trials_null)
        stats_null_arr = np.stack(all_trial_stats_list, axis=1)

        if return_probs:
            # Stack probabilities across trials: list of (B, num_folds, M) arrays -> (B, num_trials_null, num_folds, M)
            probs_null_arr = np.stack(all_trial_probs_list, axis=1)
            return probs_null_arr, stats_null_arr
        else:
            return stats_null_arr  # Shape (B, num_trials_null)

    def p_value(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        verbosity: int = 0,
    ) -> np.ndarray:
        r"""Computes the p-value for L-C2ST for a batch of observations.

        p-value = P(T_null >= T_data | H0) ≈ (1/H) Σ I(T_null,h >= T_data)

        Args:
            theta_o: Samples from the posterior conditioned on `x_o`, shape (B, M, D_theta).
                     If `self.permutation=False`, `self.theta_o_base` is used internally
                     for null statistics calculation.
            x_o: The observation batch, shape (B, D_x).
            verbosity: Verbosity level for null hypothesis evaluation, defaults to 0.

        Returns:
            Estimated p-values for each observation, shape (B,).
        """
        # Shape: (B,)
        stat_data = self.get_statistic_on_observed_data(
            theta_o=theta_o, x_o=x_o
        )
        # Shape: (B, num_trials_null)
        stats_null = self.get_statistics_under_null_hypothesis(
            theta_o=theta_o, x_o=x_o, return_probs=False, verbosity=verbosity
        )

        # Compare each null stat with the corresponding data stat
        # stat_data needs shape (B, 1) for broadcasting against (B, num_trials_null)
        stat_data_exp = stat_data[:, np.newaxis]  # Shape (B, 1)

        # Calculate proportion of null statistics >= observed statistic for each observation
        # Mean over the trials dimension (axis=1)
        p_val = np.mean(stats_null >= stat_data_exp, axis=1)  # Shape (B,)
        return p_val

    def reject_test(
        self,
        theta_o: Tensor,
        x_o: Tensor,
        alpha: float = 0.05,
        verbosity: int = 0,
    ) -> np.ndarray:
        """Performs the L-C2ST test for a batch of observations.

        Args:
            theta_o: Samples from the posterior conditioned on `x_o`, shape (B, M, D_theta).
                     If `self.permutation=False`, `self.theta_o_base` is used internally
                     for null statistics calculation.
            x_o: The observation batch, shape (B, D_x).
            alpha: Significance level, defaults to 0.05.
            verbosity: Verbosity level for p-value calculation, defaults to 0.

        Returns:
            Boolean array indicating rejection for each observation, shape (B,).
            True if the null hypothesis is rejected (p-value < alpha), False otherwise.
        """
        # p_val shape: (B,)
        p_val = self.p_value(theta_o=theta_o, x_o=x_o, verbosity=verbosity)
        # Returns boolean array of shape (B,)
        return p_val < alpha


# --- LC2ST_NF Subclass ---


class LC2ST_NF(LC2ST):
    def __init__(
        self,
        thetas: Union[np.ndarray, Tensor],
        xs: Union[np.ndarray, Tensor],
        posterior_samples: Union[np.ndarray, Tensor],
        flow_inverse_transform: Callable[[Tensor, Tensor], Tensor],
        flow_base_dist: torch.distributions.Distribution,
        num_eval: int = 10_000,  # Number of base samples for evaluation
        trained_clfs_null: Optional[
            Dict[int, List[Union[BaseEstimator, NeuralNetClassifier]]]
        ] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        L-C2ST for Normalizing Flows (using base distribution space).

        Transforms the problem to the base distribution space using `flow_inverse_transform`.
        The null hypothesis becomes H0: p(z|x) = base_dist, where z = T_inv(theta; x).
        The test compares samples transformed from the posterior `p(z|x)` against samples
        directly from the `flow_base_dist`.

        Args:
            thetas: Samples from the prior, shape (N, D_theta). Can be numpy array or torch tensor.
            xs: Corresponding observations, shape (N, D_x). Can be numpy array or torch tensor.
            posterior_samples: Samples from estimated posterior, shape (N, D_theta). Can be numpy array or torch tensor.
            flow_inverse_transform: Function T_inv(theta, x) -> z.
            flow_base_dist: The base distribution (e.g., standard Normal).
            num_eval: Number of samples to draw from `flow_base_dist` for evaluation steps.
            trained_clfs_null: Optional pre-trained classifiers under H0. Since H0 is
                               fixed (base_dist vs base_dist), these can be reused.
            **kwargs: Additional arguments passed to the `LC2ST` base class constructor
                      (e.g., classifier, num_folds, num_ensemble, seed, device).
        """
        print("Initializing LC2ST_NF...")
        self.flow_inverse_transform = flow_inverse_transform
        self._flow_base_dist = flow_base_dist  # Store original base dist
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Convert numpy arrays to torch tensors if needed
        if isinstance(thetas, np.ndarray):
            thetas = torch.from_numpy(thetas).float()
        if isinstance(xs, np.ndarray):
            xs = torch.from_numpy(xs).float()
        if isinstance(posterior_samples, np.ndarray):
            posterior_samples = torch.from_numpy(posterior_samples).float()
        # Transform thetas (reference q) and posterior samples (target p) to base space
        print("Transforming samples to base space...")
        with torch.no_grad():
            # The flow_inverse_transform function assume numpy arrays as input
            thetas_dev = thetas.to(self.device)
            xs_dev = xs.to(self.device)
            posterior_samples_dev = posterior_samples.to(self.device)

            thetas_base = flow_inverse_transform(thetas, xs).detach()
            posterior_samples_base = flow_inverse_transform(
                posterior_samples_dev, xs_dev
            ).detach()
            xs = xs_dev  # Keep xs on device

        print("Initializing base LC2ST class...")
        kwargs["permutation"] = False
        super().__init__(
            thetas=thetas_base,
            xs=xs,
            posterior_samples=posterior_samples_base,
            device=self.device,
            **kwargs,
        )

        self.null_distribution = self._flow_base_dist
        self.trained_clfs_null_ = trained_clfs_null

        self.num_eval = num_eval
        # Store base samples used for evaluation when permutation=False
        self.theta_o_base: Tensor = self._flow_base_dist.sample(
            torch.Size([num_eval])
        ).to(self.device)

        self.use_presampled_base_for_eval = kwargs.get(
            "use_presampled_base_for_eval", False
        )  # Whether to use pre-sampled base distribution points for evaluation. If True, self.theta_o_base is used for evaluation instead of sampling from the base distribution every time.
        print("LC2ST_NF Initialization complete.")

    def get_statistic_on_observed_data(
        self,
        x_o: Union[np.ndarray, Tensor],
        **kwargs: Any,  # Only x_o needed
    ) -> np.ndarray:  # Returns array shape (B,)
        """Computes T_data using transformed posterior samples vs x_o."""
        # Convert numpy array to tensor if needed
        if isinstance(x_o, np.ndarray):
            x_o = torch.from_numpy(x_o).float()

        # The classifier (trained p_base vs q_base) is evaluated using base samples
        # (self.theta_o_base) against the observation(s) x_o.
        # This aligns with how T_null is calculated (base vs base).
        if self.trained_clfs_ is None:
            raise RuntimeError("Run `train_on_observed_data` first.")
        if self.theta_o_base is None:
            raise RuntimeError("`self.theta_o_base` not set during init.")

        # Base class get_scores handles using self.theta_o_base (M, D) with x_o (B, D)
        scores = super().get_scores(
            theta_o=self.theta_o_base,  # Shape (M, D_theta)
            x_o=x_o,  # Shape (B, D_x)
            trained_clfs=self.trained_clfs_,
            return_probs=False,
        )  # Returns shape (B, num_folds)
        return scores.mean(axis=1)  # Mean over folds -> shape (B,)

    def get_statistics_under_null_hypothesis(
        self,
        x_o: Union[np.ndarray, Tensor],  # Only x_o needed
        return_probs: bool = False,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray]
    ]:  # Returns array shape (B, num_trials) or tuple
        """Computes T_null using pre-sampled base distribution points."""
        # Convert numpy array to tensor if needed
        if isinstance(x_o, np.ndarray):
            x_o = torch.from_numpy(x_o).float()

        # Base class method already handles self.permutation=False correctly,
        # using self.theta_o_base for evaluation via _prepare_eval_data.
        # We just need to pass a dummy theta_o (or self.theta_o_base itself)
        # as the base class method expects it, even though it will use self.theta_o_base
        # internally because permutation is False.
        if self.theta_o_base is None:
            raise RuntimeError("`self.theta_o_base` not set during init.")

        return super().get_statistics_under_null_hypothesis(
            theta_o=self.theta_o_base,  # Pass base samples, base method uses these when permutation=False
            x_o=x_o,
            return_probs=return_probs,
            verbosity=verbosity,
        )

    def p_value(
        self, x_o: Union[np.ndarray, Tensor], verbosity: int = 0, **kwargs: Any
    ) -> np.ndarray:  # Returns array shape (B,)
        """Computes p-value using pre-sampled base distribution points."""
        # Convert numpy array to tensor if needed
        if isinstance(x_o, np.ndarray):
            x_o = torch.from_numpy(x_o).float()

        # Base class method calls the overridden methods above.
        # Need to pass self.theta_o_base to satisfy the base p_value signature,
        # although the overridden methods it calls only need x_o.
        if self.theta_o_base is None:
            raise RuntimeError("`self.theta_o_base` not set during init.")
        return super().p_value(
            theta_o=self.theta_o_base,  # Pass base samples
            x_o=x_o,
            verbosity=verbosity,
        )

    def reject_test(
        self,
        x_o: Union[np.ndarray, Tensor],  # Only x_o needed
        alpha: float = 0.05,
        verbosity: int = 0,
        **kwargs: Any,
    ) -> np.ndarray:  # Returns array shape (B,)
        """Performs test using pre-sampled base distribution points."""
        # Convert numpy array to tensor if needed
        if isinstance(x_o, np.ndarray):
            x_o = torch.from_numpy(x_o).float()

        # Base class method calls the overridden p_value.
        if self.theta_o_base is None:
            raise RuntimeError("`self.theta_o_base` not set during init.")
        return super().reject_test(
            theta_o=self.theta_o_base,  # Pass base samples
            x_o=x_o,
            alpha=alpha,
            verbosity=verbosity,
        )
