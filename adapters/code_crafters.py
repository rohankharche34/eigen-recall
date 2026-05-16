"""
PCAM Precision Agent
====================

Design goals
------------
1. Preserve the strong retrieval performance already achieved.
2. Preserve the current anisotropy spread reduction behaviour.
3. Improve readability, maintainability, and reproducibility for judging.

Core idea
---------
The agent combines two complementary precision strategies:

A) Retrieval precision
   - Detects reliable dimensions in the corrupted query.
   - Emphasises dimensions that help distinguish nearby memories.

B) Geometry precision
   - Precomputes per-attractor anisotropy-correcting precision vectors.
   - Reduces Hessian spectral spread through diagonal congruence scaling.

At inference time:
    final_pi =
        retrieval_only                    (low confidence)
        blended retrieval + geometry      (high confidence)

No retraining of the PCAM model is performed.
Only inference-time precision steering is used.

Humanity really invented an entire competition around selectively stretching
64-dimensional valleys so noisy digits emotionally commit to the correct hole
in phase space. Remarkable species.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


class Engine(Adapter):
    """
    PCAM precision steering agent.

    Parameters
    ----------
    stored_patterns:
        Matrix of stored attractor patterns with shape (K, N).

    model_params:
        Frozen PCAM parameters supplied by the benchmark harness.
    """

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:

        # --------------------------------------------------------------
        # Stored memories
        # --------------------------------------------------------------

        self.X = stored_patterns.astype(np.float64)

        self.K, self.N = self.X.shape

        self.eps = 1e-12

        # --------------------------------------------------------------
        # Frozen PCAM model
        # --------------------------------------------------------------

        self.model = PCAMModel(
            self.X,
            model_params.get("R", np.eye(self.N)).astype(np.float64),
            eta=float(model_params.get("eta", 0.5)),
            beta=float(model_params.get("beta", 8.0)),
            pi_min=float(model_params.get("pi_min", 0.1)),
            pi_max=float(model_params.get("pi_max", 10.0)),
        )

        self.R = self.model.R
        self.eta = self.model.eta
        self.beta = self.model.beta

        self.pi_min = self.model.pi_min
        self.pi_max = self.model.pi_max

        # --------------------------------------------------------------
        # Normalized stored patterns
        # Used for cosine retrieval
        # --------------------------------------------------------------

        self.Xn = self.X / (
            np.linalg.norm(self.X, axis=1, keepdims=True) + self.eps
        )

        # --------------------------------------------------------------
        # Precompute geometry-aware anisotropy precision vectors
        # --------------------------------------------------------------

        self.pis_aniso, self.spread_reduction = (
            self._precompute_pis()
        )

    # ------------------------------------------------------------------
    # NUMERICAL UTILITIES
    # ------------------------------------------------------------------

    def _normalize_pi(self, pi: np.ndarray) -> np.ndarray:
        """
        Clamp precision into valid range and normalize mean to 1.
        """

        pi = np.clip(pi, self.pi_min, self.pi_max)

        return pi / (pi.mean() + self.eps)

    def _sym_spread(
        self,
        pi: np.ndarray,
        H: np.ndarray,
    ) -> float:
        """
        Compute spectral spread after diagonal congruence scaling.

        Spread = lambda_max / lambda_min
        Smaller is better.
        """

        sqrt_pi = np.sqrt(np.maximum(pi, self.eps))

        S = (
            sqrt_pi[:, None]
            * H
        ) * sqrt_pi[None, :]

        S = 0.5 * (S + S.T)

        eigvals = np.linalg.eigvalsh(S)

        eigvals = eigvals[eigvals > 1e-9]

        if eigvals.size < 2:
            return float("inf")

        return float(eigvals[-1] / eigvals[0])

    # ------------------------------------------------------------------
    # ANISOTROPY OPTIMIZATION
    # ------------------------------------------------------------------

    def _optimise_pi(
        self,
        H: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize diagonal precision vector for Hessian isotropisation.

        Objective:
            Minimize condition number of:
                sqrt(Pi) H sqrt(Pi)

        Returns
        -------
        best_pi:
            Best diagonal precision vector found.

        reduction:
            Spread reduction factor relative to identity precision.
        """

        n = self.N

        H = 0.5 * (H + H.T)

        eigvals_H = np.linalg.eigvalsh(H)

        valid = eigvals_H > self.eps

        if valid.sum() < 2:
            return np.ones(n, dtype=np.float64), 1.0

        base_spread = (
            eigvals_H[valid][-1]
            / eigvals_H[valid][0]
        )

        best_pi = np.ones(n, dtype=np.float64)

        best_spread = base_spread

        # --------------------------------------------------------------
        # Eigen decomposition for initialization heuristics
        # --------------------------------------------------------------

        eigvals, eigvecs = np.linalg.eigh(H)

        init_pis: list[np.ndarray] = []

        # --------------------------------------------------------------
        # Piecewise spectral initializations
        # --------------------------------------------------------------

        for n_groups in [4, 8, 12]:

            weakest_vec_sq = eigvecs[:, 0] ** 2

            order = np.argsort(weakest_vec_sq)

            groups = np.array_split(order, n_groups)

            pi_piecewise = np.ones(n, dtype=np.float64)

            for group_idx, group in enumerate(groups):

                pi_piecewise[group] = (
                    0.1
                    + 3.0 * group_idx / (n_groups - 1)
                )

            pi_piecewise = self._normalize_pi(pi_piecewise)

            init_pis.append(pi_piecewise)

        # --------------------------------------------------------------
        # Jacobi-style diagonal preconditioner
        # --------------------------------------------------------------

        h_diag = np.abs(np.diag(H))

        h_diag = np.maximum(h_diag, 1e-4)

        pi_jacobi = 1.0 / np.sqrt(h_diag)

        pi_jacobi = pi_jacobi / (
            pi_jacobi.mean() + self.eps
        )

        init_pis.append(pi_jacobi)

        # Identity baseline
        init_pis.append(np.ones(n))

        # --------------------------------------------------------------
        # Gradient optimization in log-space
        # --------------------------------------------------------------

        for init in init_pis:

            log_pi = np.log(np.maximum(init, 1e-8))

            momentum = np.zeros(n)

            learning_rate = 0.2

            stall_counter = 0

            for step_idx in range(3000):

                pi = np.exp(log_pi)

                pi = self._normalize_pi(pi)

                sqrt_pi = np.sqrt(pi)

                S = (
                    sqrt_pi[:, None]
                    * H
                ) * sqrt_pi[None, :]

                S = 0.5 * (S + S.T)

                eigvals_S, eigvecs_S = np.linalg.eigh(S)

                valid = eigvals_S > 1e-9

                if valid.sum() < 2:
                    break

                vals = eigvals_S[valid]

                current_spread = vals[-1] / vals[0]

                # --------------------------------------------------
                # Track best solution
                # --------------------------------------------------

                if current_spread < best_spread:

                    best_spread = current_spread

                    best_pi = pi.copy()

                    stall_counter = 0

                else:
                    stall_counter += 1

                # Early stopping
                if stall_counter > 300:
                    break

                if current_spread <= base_spread / 1.8:
                    break

                if (
                    step_idx > 200
                    and abs(current_spread - best_spread)
                    / (best_spread + self.eps)
                    < 1e-10
                ):
                    break

                # --------------------------------------------------
                # Spectral gradient
                # --------------------------------------------------

                strongest_vec = eigvecs_S[:, valid][:, -1]

                weakest_vec = eigvecs_S[:, valid][:, 0]

                gradient = (
                    strongest_vec ** 2
                    - weakest_vec ** 2
                )

                gradient -= gradient.mean()

                grad_norm = np.linalg.norm(gradient)

                if grad_norm < self.eps:
                    break

                momentum = (
                    0.8 * momentum
                    + 0.2 * (gradient / grad_norm)
                )

                momentum_norm = np.linalg.norm(momentum)

                if momentum_norm < self.eps:
                    break

                adaptive_step = (
                    learning_rate
                    / (1.0 + 0.003 * step_idx)
                )

                log_pi -= (
                    adaptive_step
                    * momentum
                    / momentum_norm
                )

        best_pi = self._normalize_pi(best_pi)

        final_spread = self._sym_spread(best_pi, H)

        reduction = (
            base_spread / final_spread
            if final_spread > self.eps
            else 1.0
        )

        return best_pi, reduction

    # ------------------------------------------------------------------
    # PRECOMPUTATION
    # ------------------------------------------------------------------

    def _precompute_pis(
        self,
    ) -> Tuple[np.ndarray, float]:
        """
        Precompute anisotropy-correcting precision vectors
        for every stored attractor.
        """

        pis = np.empty(
            (self.K, self.N),
            dtype=np.float64,
        )

        reductions = np.empty(
            self.K,
            dtype=np.float64,
        )

        for idx in range(self.K):

            attractor = self.model.find_equilibrium(
                self.X[idx]
            )

            H = self.model.hessian(attractor)

            (
                pis[idx],
                reductions[idx],
            ) = self._optimise_pi(H)

        return pis, reductions.mean()

    # ------------------------------------------------------------------
    # RETRIEVAL PRECISION
    # ------------------------------------------------------------------

    def _retrieval_pi(
        self,
        q: np.ndarray,
        q_unit: np.ndarray,
        cosines: np.ndarray,
        max_cos: float,
    ) -> np.ndarray:
        """
        Construct retrieval-oriented precision.

        Components
        ----------
        1. Reliability weighting
           Downweights corrupted dimensions.

        2. Discriminative weighting
           Upweights dimensions that separate nearby memories.
        """

        sims_shifted = cosines - max_cos

        weights = np.exp(
            self.beta * sims_shifted
        )

        weights /= (
            weights.sum() + self.eps
        )

        # --------------------------------------------------------------
        # Expected clean reconstruction
        # --------------------------------------------------------------

        x_hat = weights @ self.X

        disagreement = np.abs(q - x_hat)

        reliability = np.exp(-2.5 * disagreement)

        reliability += 0.05

        reliability /= (
            reliability.mean() + self.eps
        )

        # --------------------------------------------------------------
        # Local discriminative variance
        # --------------------------------------------------------------

        top_k = min(4, self.K)

        top_idx = np.argpartition(
            cosines,
            -top_k,
        )[-top_k:]

        top_weights = (
            weights[top_idx] + self.eps
        )

        top_weights /= top_weights.sum()

        mean_top = top_weights @ self.X[top_idx]

        variance_top = top_weights @ (
            (self.X[top_idx] - mean_top) ** 2
        )

        discriminative = variance_top / (
            variance_top.mean() + self.eps
        )

        discriminative /= (
            discriminative.mean() + self.eps
        )

        # --------------------------------------------------------------
        # Combined retrieval precision
        # --------------------------------------------------------------

        pi = reliability * (
            0.3 + 0.7 * discriminative
        )

        return self._normalize_pi(pi)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def predict_precision(
        self,
        corrupted_query: np.ndarray,
    ) -> np.ndarray:
        """
        Predict a 64-dimensional positive precision vector.
        """

        q = np.asarray(
            corrupted_query,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        if (
            q.ndim != 1
            or q.shape[0] != self.N
            or not np.all(np.isfinite(q))
        ):
            return np.ones(
                self.N,
                dtype=np.float64,
            )

        q_norm = np.linalg.norm(q)

        if q_norm < self.eps:
            return np.ones(
                self.N,
                dtype=np.float64,
            )

        # --------------------------------------------------------------
        # Retrieval confidence
        # --------------------------------------------------------------

        q_unit = q / q_norm

        cosines = self.Xn @ q_unit

        max_cos = cosines.max()

        best_idx = cosines.argmax()

        # --------------------------------------------------------------
        # Retrieval precision
        # --------------------------------------------------------------

        pi_retrieval = self._retrieval_pi(
            q,
            q_unit,
            cosines,
            max_cos,
        )

        # --------------------------------------------------------------
        # High-confidence geometry blending
        # --------------------------------------------------------------

        if max_cos > 0.85:

            pi_aniso = self.pis_aniso[best_idx]

            # Keep original scoring behaviour intact.
            # Do NOT alter these weights unless re-tuning.
            pi = (
                0.1 * pi_retrieval
                + 0.9 * pi_aniso
            )

            return self._normalize_pi(pi)

        return pi_retrieval
