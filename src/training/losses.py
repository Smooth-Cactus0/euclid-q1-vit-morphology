"""Loss functions for galaxy morphology regression.

The primary loss is Dirichlet-Multinomial, matching Zoobot (Walmsley et al. 2025).
A masked MSE fallback is also provided for comparison and debugging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import MorphologySchema


class DirichletMultinomialLoss(nn.Module):
    """Dirichlet-Multinomial negative log-likelihood loss.

    For each morphology question, the model predicts concentration parameters
    of a Dirichlet distribution over answer fractions.  The loss measures how
    well those concentrations explain the observed vote fractions.

    This matches the Zoobot training objective, enabling direct comparison.

    The model outputs raw logits which are passed through softplus to get
    positive concentrations per question group.

    Parameters
    ----------
    schema : MorphologySchema
        Defines which output indices belong to each question.
    total_count : float
        Pseudo-count representing total number of "votes" per question.
        Higher values make the loss more sensitive to small differences.
        Zoobot uses values around 30-40.
    """

    def __init__(self, schema: MorphologySchema, total_count: float = 40.0) -> None:
        super().__init__()
        self.schema = schema
        self.total_count = total_count

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked Dirichlet-Multinomial loss.

        Parameters
        ----------
        predictions : [B, num_outputs] raw logits from model
        targets : [B, num_outputs] vote fractions (0 where masked)
        masks : [B, num_outputs] 1.0 where valid, 0.0 where NaN

        Returns
        -------
        Scalar loss (mean over valid questions and batch).
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        n_valid = 0

        for question, (start, end) in self.schema.question_slices.items():
            # Per-question mask: valid if any answer in the group is valid
            q_mask = masks[:, start] > 0  # [B] boolean

            if q_mask.sum() == 0:
                continue

            # Get predictions and targets for valid samples only
            q_pred = predictions[q_mask, start:end]   # [B_valid, n_answers]
            q_target = targets[q_mask, start:end]      # [B_valid, n_answers]

            # Softplus to get positive concentrations
            concentrations = F.softplus(q_pred) + 1e-6  # [B_valid, n_answers]

            # Scale concentrations so they sum to total_count
            conc_sum = concentrations.sum(dim=1, keepdim=True)
            concentrations = concentrations / conc_sum * self.total_count

            # Pseudo-counts from observed fractions
            observed_counts = q_target * self.total_count

            # Dirichlet-Multinomial NLL
            # log B(observed + alpha) - log B(alpha) + log C(N, observed)
            # We drop the combinatorial term (constant w.r.t. parameters)
            loss = (
                torch.lgamma(concentrations.sum(dim=1))
                - torch.lgamma(concentrations.sum(dim=1) + self.total_count)
                + (
                    torch.lgamma(concentrations + observed_counts)
                    - torch.lgamma(concentrations)
                ).sum(dim=1)
            )

            # Negate (we computed log-likelihood, want NLL)
            total_loss = total_loss - loss.mean()
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return total_loss / n_valid


class MaskedMSELoss(nn.Module):
    """Masked MSE loss for multi-task regression with missing targets.

    Computes MSE only on valid (non-NaN) outputs.  Simpler than
    Dirichlet-Multinomial, useful as a sanity check and fallback.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : [B, num_outputs] — model outputs (after sigmoid for fractions)
        targets : [B, num_outputs] — vote fractions
        masks : [B, num_outputs] — validity mask

        Returns
        -------
        Scalar masked MSE loss.
        """
        # Apply sigmoid to get [0, 1] range for MSE variant
        pred_fracs = torch.sigmoid(predictions)
        sq_error = (pred_fracs - targets) ** 2
        masked_error = sq_error * masks

        n_valid = masks.sum()
        if n_valid == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return masked_error.sum() / n_valid
