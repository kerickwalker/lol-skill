"""Baseline plus direct gameplay-rule constraints.

This model intentionally builds on ``models.baseline`` and only adds feature
relationships labeled ``directly_increases`` in
``feature-relationships/simple/lck_s15_feature_relationship_edges_simplified.csv``.
Other edge types such as ``component_of`` and ``precondition`` are ignored here.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models import baseline
from models.baseline import (  # re-exported for train.py/case_study.py compatibility
    DIFF_STATS,
    TEAM_CONTEXT_STATS,
    build_player_score_table,
    load_data,
    print_rankings,
    print_score_table,
)
from models.config import INDIVIDUAL_STATS


USE_SCORE_TABLE_AS_MAIN_OUTPUT = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULE_EDGES_PATH = (
    PROJECT_ROOT
    / "feature-relationships"
    / "simple"
    / "lck_s15_feature_relationship_edges_simplified.csv"
)

RULE_WEIGHT_PRIOR_CENTER = 0.35
RULE_WEIGHT_PRIOR_LOG_STD = 0.75
RULE_TAU_PRIOR_CENTER = 1.0
RULE_TAU_PRIOR_LOG_STD = 0.35
RULE_TAU_FLOOR = 0.05


def _load_direct_rules():
    edges = pd.read_csv(RULE_EDGES_PATH)
    edges = edges[edges["kind"] == "directly_increases"].copy()

    individual: dict[str, list[str]] = {}
    diff: dict[str, list[str]] = {}
    team: dict[str, list[str]] = {}

    for row in edges.itertuples(index=False):
        source = row.source
        target = row.target
        if source in INDIVIDUAL_STATS and target in INDIVIDUAL_STATS:
            individual.setdefault(target, []).append(source)
        elif source in DIFF_STATS and target in DIFF_STATS:
            diff.setdefault(target, []).append(source)
        elif source in TEAM_CONTEXT_STATS and target in TEAM_CONTEXT_STATS:
            team.setdefault(target, []).append(source)
        else:
            raise ValueError(
                "Direct rule edge cannot be mapped to known feature groups: "
                f"{source!r} -> {target!r}"
            )

    return individual, diff, team


INDIVIDUAL_RULES, DIFF_RULES, TEAM_RULES = _load_direct_rules()
N_INDIVIDUAL_RULE_WEIGHTS = sum(len(sources) for sources in INDIVIDUAL_RULES.values())
N_DIFF_RULE_WEIGHTS = sum(len(sources) for sources in DIFF_RULES.values())
N_TEAM_RULE_WEIGHTS = sum(len(sources) for sources in TEAM_RULES.values())
N_RULE_GROUPS = len(INDIVIDUAL_RULES) + len(DIFF_RULES) + len(TEAM_RULES)


def _lognormal_sample(name: str, size: int, center: float, log_std: float):
    if size == 0:
        return torch.empty(0)
    loc = torch.log(center * torch.ones(size, dtype=torch.float32))
    scale = log_std * torch.ones(size, dtype=torch.float32)
    return pyro.sample(name, dist.LogNormal(loc, scale).to_event(1))


def _lognormal_guide(name: str, size: int, center: float):
    if size == 0:
        return
    loc = pyro.param(f"{name}_loc", torch.log(center * torch.ones(size, dtype=torch.float32)))
    scale = pyro.param(
        f"{name}_scale",
        0.1 * torch.ones(size, dtype=torch.float32),
        constraint=dist.constraints.positive,
    )
    pyro.sample(name, dist.LogNormal(loc, scale).to_event(1))


def _positive_rule_weights():
    return {
        "individual": _lognormal_sample(
            "rule_individual_weight",
            N_INDIVIDUAL_RULE_WEIGHTS,
            RULE_WEIGHT_PRIOR_CENTER,
            RULE_WEIGHT_PRIOR_LOG_STD,
        ),
        "diff": _lognormal_sample(
            "rule_diff_weight",
            N_DIFF_RULE_WEIGHTS,
            RULE_WEIGHT_PRIOR_CENTER,
            RULE_WEIGHT_PRIOR_LOG_STD,
        ),
        "team": _lognormal_sample(
            "rule_team_weight",
            N_TEAM_RULE_WEIGHTS,
            RULE_WEIGHT_PRIOR_CENTER,
            RULE_WEIGHT_PRIOR_LOG_STD,
        ),
        "tau": _lognormal_sample(
            "rule_tau",
            N_RULE_GROUPS,
            RULE_TAU_PRIOR_CENTER,
            RULE_TAU_PRIOR_LOG_STD,
        )
        + RULE_TAU_FLOOR,
    }


def _rule_guide():
    _lognormal_guide("rule_individual_weight", N_INDIVIDUAL_RULE_WEIGHTS, RULE_WEIGHT_PRIOR_CENTER)
    _lognormal_guide("rule_diff_weight", N_DIFF_RULE_WEIGHTS, RULE_WEIGHT_PRIOR_CENTER)
    _lognormal_guide("rule_team_weight", N_TEAM_RULE_WEIGHTS, RULE_WEIGHT_PRIOR_CENTER)
    _lognormal_guide("rule_tau", N_RULE_GROUPS, RULE_TAU_PRIOR_CENTER)


def _add_rule_likelihoods(batch, weights):
    weight_offsets = {"individual": 0, "diff": 0, "team": 0}
    tau_offset = 0

    for target, sources in INDIVIDUAL_RULES.items():
        target_idx = INDIVIDUAL_STATS.index(target)
        source_idx = torch.tensor([INDIVIDUAL_STATS.index(source) for source in sources], dtype=torch.long)
        start = weight_offsets["individual"]
        end = start + len(sources)
        weight_offsets["individual"] = end
        rule_weights = weights["individual"][start:end]
        tau = weights["tau"][tau_offset]
        tau_offset += 1

        source_a = batch["stats_a"][..., source_idx]
        source_b = batch["stats_b"][..., source_idx]
        mean_a = (source_a * rule_weights).sum(-1)
        mean_b = (source_b * rule_weights).sum(-1)
        obs_a = batch["stats_a"][..., target_idx]
        obs_b = batch["stats_b"][..., target_idx]
        pyro.sample(
            f"rule_individual_{target}_a",
            dist.Normal(mean_a, tau).to_event(2),
            obs=obs_a,
        )
        pyro.sample(
            f"rule_individual_{target}_b",
            dist.Normal(mean_b, tau).to_event(2),
            obs=obs_b,
        )

    for target, sources in DIFF_RULES.items():
        target_idx = DIFF_STATS.index(target)
        source_idx = torch.tensor([DIFF_STATS.index(source) for source in sources], dtype=torch.long)
        start = weight_offsets["diff"]
        end = start + len(sources)
        weight_offsets["diff"] = end
        rule_weights = weights["diff"][start:end]
        tau = weights["tau"][tau_offset]
        tau_offset += 1

        source_a = batch["diff_stats_a"][..., source_idx]
        source_b = batch["diff_stats_b"][..., source_idx]
        mean_a = (source_a * rule_weights).sum(-1)
        mean_b = (source_b * rule_weights).sum(-1)
        obs_a = batch["diff_stats_a"][..., target_idx]
        obs_b = batch["diff_stats_b"][..., target_idx]
        pyro.sample(
            f"rule_diff_{target}_a",
            dist.Normal(mean_a, tau).to_event(2),
            obs=obs_a,
        )
        pyro.sample(
            f"rule_diff_{target}_b",
            dist.Normal(mean_b, tau).to_event(2),
            obs=obs_b,
        )

    for target, sources in TEAM_RULES.items():
        target_idx = TEAM_CONTEXT_STATS.index(target)
        source_idx = torch.tensor([TEAM_CONTEXT_STATS.index(source) for source in sources], dtype=torch.long)
        start = weight_offsets["team"]
        end = start + len(sources)
        weight_offsets["team"] = end
        rule_weights = weights["team"][start:end]
        tau = weights["tau"][tau_offset]
        tau_offset += 1

        source_a = batch["team_output_context_a"][..., source_idx]
        source_b = batch["team_output_context_b"][..., source_idx]
        mean_a = (source_a * rule_weights).sum(-1)
        mean_b = (source_b * rule_weights).sum(-1)
        obs_a = batch["team_output_context_a"][..., target_idx]
        obs_b = batch["team_output_context_b"][..., target_idx]
        pyro.sample(
            f"rule_team_{target}_a",
            dist.Normal(mean_a, tau).to_event(1),
            obs=obs_a,
        )
        pyro.sample(
            f"rule_team_{target}_b",
            dist.Normal(mean_b, tau).to_event(1),
            obs=obs_b,
        )


def model(batch, n_players):
    baseline.model(batch, n_players)
    weights = _positive_rule_weights()
    _add_rule_likelihoods(batch, weights)


def guide(batch, n_players):
    baseline.guide(batch, n_players)
    _rule_guide()


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
