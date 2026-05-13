"""Baseline with category-informed role-specific stat effects and noise.

This model builds directly on ``models.baseline``. It keeps the same data
loading, z-scoring, context effects, skill prior, and result model, but replaces
shared ``alpha_stat``/``tau_stat`` with role-specific versions.

For selected core stats, prior means/noise centers encode explicit domain
categories discussed in the project notes. All values are still learned.
"""

from __future__ import annotations

import os

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models import baseline
from models.baseline import (
    DIFF_STATS,
    DURATION_CONTEXT_STATS,
    TEAM_CONTEXT_STATS,
    build_player_score_table,
    load_data,
    print_rankings,
    print_score_table,
)
from models.config import INDIVIDUAL_STATS, N_ROLES


SKILL_PRIOR_MEAN = baseline.SKILL_PRIOR_MEAN
SKILL_PRIOR_STD = baseline.SKILL_PRIOR_STD
PERFORMANCE_BETA = baseline.PERFORMANCE_BETA
RESULT_NOISE = baseline.RESULT_NOISE
ALPHA_PRIOR_STD = baseline.ALPHA_PRIOR_STD
TAU_PRIOR_CENTER = baseline.TAU_PRIOR_CENTER
TAU_PRIOR_LOG_STD = baseline.TAU_PRIOR_LOG_STD
TAU_FLOOR = baseline.TAU_FLOOR
CONTEXT_PRIOR_STD = baseline.CONTEXT_PRIOR_STD
USE_SCORE_TABLE_AS_MAIN_OUTPUT = True

ALPHA_ROLE_DEVIATION_STD = float(os.environ.get("LCK_ALPHA_ROLE_DEVIATION_STD", "0.35"))
TAU_ROLE_LOG_DEVIATION_STD = float(os.environ.get("LCK_TAU_ROLE_LOG_DEVIATION_STD", "0.35"))

RAW_TAU_PRIOR_CENTER = max(TAU_PRIOR_CENTER - TAU_FLOOR, 1e-3)
BASE_TAU_VEC = RAW_TAU_PRIOR_CENTER * torch.ones(len(INDIVIDUAL_STATS), dtype=torch.float32)
BASE_DIFF_TAU_VEC = RAW_TAU_PRIOR_CENTER * torch.ones(len(DIFF_STATS), dtype=torch.float32)
ALPHA_PRIOR_SCALE = ALPHA_PRIOR_STD * torch.ones(len(INDIVIDUAL_STATS), dtype=torch.float32)
DIFF_ALPHA_PRIOR_SCALE = ALPHA_PRIOR_STD * torch.ones(len(DIFF_STATS), dtype=torch.float32)

ALPHA_CATEGORY = {
    "very_low": (0.00, 0.20),
    "low": (0.15, 0.25),
    "medium": (0.35, 0.35),
    "high": (0.65, 0.35),
    "very_high": (0.80, 0.40),
}
TAU_CATEGORY = {
    "very_small": 0.35,
    "small": 0.60,
    "medium": 0.90,
    "high": 1.30,
    "very_high": 1.70,
}

ROLE_STAT_CATEGORIES = {
    "kills": {
        "top": ("medium", "small"),
        "jng": ("medium", "small"),
        "mid": ("very_high", "very_small"),
        "adc": ("very_high", "very_small"),
        "sup": ("very_low", "high"),
    },
    "deaths": {
        "top": ("medium", "small"),
        "jng": ("medium", "small"),
        "mid": ("high", "very_small"),
        "adc": ("high", "very_small"),
        "sup": ("very_low", "high"),
    },
    "cs": {
        "top": ("high", "very_small"),
        "jng": ("medium", "medium"),
        "mid": ("high", "very_small"),
        "adc": ("high", "very_small"),
        "sup": ("very_low", "very_high"),
    },
    "golds": {
        "top": ("very_high", "very_small"),
        "jng": ("medium", "medium"),
        "mid": ("very_high", "very_small"),
        "adc": ("very_high", "very_small"),
        "sup": ("very_low", "very_high"),
    },
}
NEGATIVE_ALPHA_STATS = {"deaths"}


def _build_role_stat_priors():
    alpha_loc = torch.zeros(N_ROLES, len(INDIVIDUAL_STATS), dtype=torch.float32)
    alpha_scale = ALPHA_PRIOR_STD * torch.ones(N_ROLES, len(INDIVIDUAL_STATS), dtype=torch.float32)
    tau_center = TAU_PRIOR_CENTER * torch.ones(N_ROLES, len(INDIVIDUAL_STATS), dtype=torch.float32)

    role_to_idx = {"top": 0, "jng": 1, "mid": 2, "adc": 3, "sup": 4}
    for stat, role_categories in ROLE_STAT_CATEGORIES.items():
        stat_idx = INDIVIDUAL_STATS.index(stat)
        sign = -1.0 if stat in NEGATIVE_ALPHA_STATS else 1.0
        for role, (alpha_category, tau_category) in role_categories.items():
            role_idx = role_to_idx[role]
            alpha_mean, alpha_std = ALPHA_CATEGORY[alpha_category]
            alpha_loc[role_idx, stat_idx] = sign * alpha_mean
            alpha_scale[role_idx, stat_idx] = alpha_std
            tau_center[role_idx, stat_idx] = TAU_CATEGORY[tau_category]

    tau_raw_center = torch.clamp(tau_center - TAU_FLOOR, min=1e-3)
    return alpha_loc, alpha_scale, tau_raw_center


ROLE_ALPHA_PRIOR_LOC, ROLE_ALPHA_PRIOR_SCALE, ROLE_TAU_RAW_PRIOR_CENTER = _build_role_stat_priors()
DIFF_ALPHA_PRIOR_LOC = torch.zeros(N_ROLES, len(DIFF_STATS), dtype=torch.float32)
DIFF_ALPHA_PRIOR_SCALE_MAT = ALPHA_PRIOR_STD * torch.ones(N_ROLES, len(DIFF_STATS), dtype=torch.float32)
DIFF_TAU_RAW_PRIOR_CENTER = RAW_TAU_PRIOR_CENTER * torch.ones(N_ROLES, len(DIFF_STATS), dtype=torch.float32)


def _role_gather(matrix: torch.Tensor, role_idx: torch.Tensor) -> torch.Tensor:
    return matrix[role_idx]


def model(batch, n_players):
    n_matches = batch["winner"].shape[0]

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
        p_a = pyro.sample("p_a", dist.Normal(skill_a, PERFORMANCE_BETA).to_event(1))
        p_b = pyro.sample("p_b", dist.Normal(skill_b, PERFORMANCE_BETA).to_event(1))

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
    n_matches = batch["winner"].shape[0]

    s_loc = pyro.param("s_loc", SKILL_PRIOR_MEAN * torch.ones(n_players, N_ROLES))
    s_scale = pyro.param(
        "s_scale",
        SKILL_PRIOR_STD * torch.ones(n_players, N_ROLES),
        constraint=dist.constraints.positive,
    )
    pyro.sample("s", dist.Normal(s_loc, s_scale).to_event(2))

    alpha_role_loc = pyro.param("alpha_role_loc", ROLE_ALPHA_PRIOR_LOC.clone())
    alpha_role_scale = pyro.param(
        "alpha_role_scale",
        0.1 * ROLE_ALPHA_PRIOR_SCALE,
        constraint=dist.constraints.positive,
    )
    pyro.sample("alpha_role", dist.Normal(alpha_role_loc, alpha_role_scale).to_event(2))

    tau_role_log_loc = pyro.param("tau_role_log_loc", torch.log(ROLE_TAU_RAW_PRIOR_CENTER))
    tau_role_log_scale = pyro.param(
        "tau_role_log_scale",
        0.1 * torch.ones_like(ROLE_TAU_RAW_PRIOR_CENTER),
        constraint=dist.constraints.positive,
    )
    pyro.sample("tau_role_raw", dist.LogNormal(tau_role_log_loc, tau_role_log_scale).to_event(2))

    diff_alpha_role_loc = pyro.param("diff_alpha_role_loc", DIFF_ALPHA_PRIOR_LOC.clone())
    diff_alpha_role_scale = pyro.param(
        "diff_alpha_role_scale",
        0.1 * DIFF_ALPHA_PRIOR_SCALE_MAT,
        constraint=dist.constraints.positive,
    )
    pyro.sample("diff_alpha_role", dist.Normal(diff_alpha_role_loc, diff_alpha_role_scale).to_event(2))

    diff_tau_role_log_loc = pyro.param("diff_tau_role_log_loc", torch.log(DIFF_TAU_RAW_PRIOR_CENTER))
    diff_tau_role_log_scale = pyro.param(
        "diff_tau_role_log_scale",
        0.1 * torch.ones_like(DIFF_TAU_RAW_PRIOR_CENTER),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        "diff_tau_role_raw",
        dist.LogNormal(diff_tau_role_log_loc, diff_tau_role_log_scale).to_event(2),
    )

    duration_loc = pyro.param(
        "duration_effect_loc",
        torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    duration_scale = pyro.param(
        "duration_effect_scale",
        0.1 * torch.ones(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("duration_effect", dist.Normal(duration_loc, duration_scale).to_event(2))

    team_output_loc = pyro.param(
        "team_output_effect_loc",
        torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    team_output_scale = pyro.param(
        "team_output_effect_scale",
        0.1 * torch.ones(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("team_output_effect", dist.Normal(team_output_loc, team_output_scale).to_event(2))

    pa_loc = pyro.param("pa_loc", SKILL_PRIOR_MEAN * torch.ones(n_matches, 5))
    pa_scale = pyro.param(
        "pa_scale",
        PERFORMANCE_BETA * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )
    pb_loc = pyro.param("pb_loc", SKILL_PRIOR_MEAN * torch.ones(n_matches, 5))
    pb_scale = pyro.param(
        "pb_scale",
        PERFORMANCE_BETA * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("matches", n_matches):
        pyro.sample("p_a", dist.Normal(pa_loc, pa_scale).to_event(1))
        pyro.sample("p_b", dist.Normal(pb_loc, pb_scale).to_event(1))


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
