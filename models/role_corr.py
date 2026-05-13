"""Role-specific alpha/tau model plus learned teammate-performance covariance.

This builds on ``models.role_alpha_tau`` and changes only the per-game
performance layer: teammate performances are sampled jointly from a
MultivariateNormal whose Cholesky factor is learned around the approved
role-influence prior. Learning the Cholesky factor keeps the covariance
positive definite.
"""

from __future__ import annotations

import os

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models import role_alpha_tau
from models.role_alpha_tau import (
    CONTEXT_PRIOR_STD,
    DIFF_ALPHA_PRIOR_LOC,
    DIFF_ALPHA_PRIOR_SCALE_MAT,
    DIFF_STATS,
    DIFF_TAU_RAW_PRIOR_CENTER,
    DURATION_CONTEXT_STATS,
    INDIVIDUAL_STATS,
    N_ROLES,
    PERFORMANCE_BETA,
    RESULT_NOISE,
    ROLE_ALPHA_PRIOR_LOC,
    ROLE_ALPHA_PRIOR_SCALE,
    ROLE_TAU_RAW_PRIOR_CENTER,
    SKILL_PRIOR_MEAN,
    SKILL_PRIOR_STD,
    TAU_FLOOR,
    TAU_PRIOR_LOG_STD,
    TEAM_CONTEXT_STATS,
    _role_gather,
    build_player_score_table,
    load_data,
    print_rankings,
    print_score_table,
)


USE_SCORE_TABLE_AS_MAIN_OUTPUT = True

ROLE_INFLUENCE = torch.tensor(
    [
        [0.00, 0.30, 0.15, 0.05, 0.10],
        [0.30, 0.00, 0.60, 0.40, 0.80],
        [0.15, 0.60, 0.00, 0.20, 0.50],
        [0.05, 0.40, 0.20, 0.00, 1.00],
        [0.10, 0.80, 0.50, 1.00, 0.00],
    ],
    dtype=torch.float32,
)
ROLE_CORR_SCALE = float(os.environ.get("LCK_ROLE_CORR_SCALE", "0.8"))
ROLE_CORR = torch.eye(N_ROLES, dtype=torch.float32) + ROLE_CORR_SCALE * ROLE_INFLUENCE
TARGET_PERF_COV = (PERFORMANCE_BETA**2) * ROLE_CORR
TARGET_SCALE_TRIL = torch.linalg.cholesky(TARGET_PERF_COV)
LOWER_TRIANGLE = torch.tril_indices(N_ROLES, N_ROLES, offset=-1)

CHOLESKY_OFFDIAG_PRIOR_STD = float(os.environ.get("LCK_ROLE_CORR_OFFDIAG_STD", "0.08"))
CHOLESKY_DIAG_LOG_PRIOR_STD = float(os.environ.get("LCK_ROLE_CORR_DIAG_LOG_STD", "0.05"))


def build_scale_tril(diag: torch.Tensor, offdiag: torch.Tensor) -> torch.Tensor:
    scale_tril = torch.diag(diag)
    scale_tril[LOWER_TRIANGLE[0], LOWER_TRIANGLE[1]] = offdiag
    return scale_tril


def sample_scale_tril() -> torch.Tensor:
    target_diag = torch.diag(TARGET_SCALE_TRIL)
    target_offdiag = TARGET_SCALE_TRIL[LOWER_TRIANGLE[0], LOWER_TRIANGLE[1]]
    diag = pyro.sample(
        "role_corr_cholesky_diag",
        dist.LogNormal(
            torch.log(target_diag),
            CHOLESKY_DIAG_LOG_PRIOR_STD * torch.ones_like(target_diag),
        ).to_event(1),
    )
    offdiag = pyro.sample(
        "role_corr_cholesky_offdiag",
        dist.Normal(
            target_offdiag,
            CHOLESKY_OFFDIAG_PRIOR_STD * torch.ones_like(target_offdiag),
        ).to_event(1),
    )
    return build_scale_tril(diag, offdiag)


def guide_scale_tril():
    target_diag = torch.diag(TARGET_SCALE_TRIL)
    target_offdiag = TARGET_SCALE_TRIL[LOWER_TRIANGLE[0], LOWER_TRIANGLE[1]]
    diag_log_loc = pyro.param("role_corr_cholesky_diag_log_loc", torch.log(target_diag))
    diag_log_scale = pyro.param(
        "role_corr_cholesky_diag_log_scale",
        0.02 * torch.ones_like(target_diag),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        "role_corr_cholesky_diag",
        dist.LogNormal(diag_log_loc, diag_log_scale).to_event(1),
    )

    offdiag_loc = pyro.param("role_corr_cholesky_offdiag_loc", target_offdiag.clone())
    offdiag_scale = pyro.param(
        "role_corr_cholesky_offdiag_scale",
        0.02 * torch.ones_like(target_offdiag),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        "role_corr_cholesky_offdiag",
        dist.Normal(offdiag_loc, offdiag_scale).to_event(1),
    )


def model(batch, n_players):
    n_matches = batch["winner"].shape[0]

    perf_scale_tril = sample_scale_tril()

    s = pyro.sample(
        "s",
        dist.Normal(SKILL_PRIOR_MEAN, SKILL_PRIOR_STD)
        .expand([n_players, N_ROLES])
        .to_event(2),
    )
    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]

    alpha_role = pyro.sample(
        "alpha_role",
        dist.Normal(ROLE_ALPHA_PRIOR_LOC, ROLE_ALPHA_PRIOR_SCALE).to_event(2),
    )
    tau_role_raw = pyro.sample(
        "tau_role_raw",
        dist.LogNormal(
            torch.log(ROLE_TAU_RAW_PRIOR_CENTER),
            TAU_PRIOR_LOG_STD * torch.ones_like(ROLE_TAU_RAW_PRIOR_CENTER),
        ).to_event(2),
    )
    tau_role = tau_role_raw + TAU_FLOOR

    diff_alpha_role = pyro.sample(
        "diff_alpha_role",
        dist.Normal(DIFF_ALPHA_PRIOR_LOC, DIFF_ALPHA_PRIOR_SCALE_MAT).to_event(2),
    )
    diff_tau_role_raw = pyro.sample(
        "diff_tau_role_raw",
        dist.LogNormal(
            torch.log(DIFF_TAU_RAW_PRIOR_CENTER),
            TAU_PRIOR_LOG_STD * torch.ones_like(DIFF_TAU_RAW_PRIOR_CENTER),
        ).to_event(2),
    )
    diff_tau_role = diff_tau_role_raw + TAU_FLOOR

    duration_effect = pyro.sample(
        "duration_effect",
        dist.Normal(
            torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            CONTEXT_PRIOR_STD,
        ).to_event(2),
    )
    team_output_effect = pyro.sample(
        "team_output_effect",
        dist.Normal(
            torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            CONTEXT_PRIOR_STD,
        ).to_event(2),
    )

    with pyro.plate("matches", n_matches):
        p_a = pyro.sample("p_a", dist.MultivariateNormal(skill_a, scale_tril=perf_scale_tril))
        p_b = pyro.sample("p_b", dist.MultivariateNormal(skill_b, scale_tril=perf_scale_tril))

        team_perf_a = p_a.mean(-1)
        team_perf_b = p_b.mean(-1)
        win_prob = dist.Normal(0.0, 1.0).cdf(
            (team_perf_a - team_perf_b) / (torch.sqrt(torch.tensor(2.0)) * RESULT_NOISE)
        )
        pyro.sample("y", dist.Bernoulli(win_prob), obs=batch["winner"])

        alpha_a = _role_gather(alpha_role, batch["team_a_role"])
        alpha_b = _role_gather(alpha_role, batch["team_b_role"])
        tau_a = _role_gather(tau_role, batch["team_a_role"])
        tau_b = _role_gather(tau_role, batch["team_b_role"])

        mean_a = p_a.unsqueeze(-1) * alpha_a
        mean_b = p_b.unsqueeze(-1) * alpha_b
        duration_shift = batch["duration_context"] @ duration_effect
        mean_a = mean_a + duration_shift.unsqueeze(1)
        mean_b = mean_b + duration_shift.unsqueeze(1)
        mean_a = mean_a + (batch["team_output_context_a"] @ team_output_effect).unsqueeze(1)
        mean_b = mean_b + (batch["team_output_context_b"] @ team_output_effect).unsqueeze(1)

        pyro.sample("obs_a", dist.Normal(mean_a, tau_a).to_event(2), obs=batch["stats_a"])
        pyro.sample("obs_b", dist.Normal(mean_b, tau_b).to_event(2), obs=batch["stats_b"])

        diff_alpha_a = _role_gather(diff_alpha_role, batch["team_a_role"])
        diff_alpha_b = _role_gather(diff_alpha_role, batch["team_b_role"])
        diff_tau_a = _role_gather(diff_tau_role, batch["team_a_role"])
        diff_tau_b = _role_gather(diff_tau_role, batch["team_b_role"])

        diff_mean_a = (p_a - p_b).unsqueeze(-1) * diff_alpha_a
        diff_mean_b = (p_b - p_a).unsqueeze(-1) * diff_alpha_b
        pyro.sample("diff_obs_a", dist.Normal(diff_mean_a, diff_tau_a).to_event(2), obs=batch["diff_stats_a"])
        pyro.sample("diff_obs_b", dist.Normal(diff_mean_b, diff_tau_b).to_event(2), obs=batch["diff_stats_b"])


def guide(batch, n_players):
    role_alpha_tau.guide(batch, n_players)
    guide_scale_tril()


def train(batch, n_players, n_steps=1500, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    disable_progress = os.environ.get("LCK_DISABLE_TQDM", "0") == "1"
    with trange(n_steps, desc="Training", unit="step", disable=disable_progress) as bar:
        for _ in bar:
            loss = svi.step(batch, n_players)
            losses.append(loss)
            if not disable_progress:
                bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses
