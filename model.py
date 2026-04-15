"""
Simplified TrueSkill-style model for 5v5 League of Legends in Pyro.

Model (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)   [latent, persistent]
    Layer 2 — per-game performance:     p_i     ~ N(s_{i,r_i}, beta^2)
    Layer 3 — team sum:                 t_A = sum_{i in A} p_i         [deterministic]
    Layer 4 — match outcome:            y   ~ Bernoulli(Phi((t_A - t_B)/sqrt(2)*beta_o))
    Layer 5 — per-player KDA stat:      KDA_i ~ N(alpha * p_i + gamma_{r_i}, tau^2)

We use a Bernoulli with a probit link instead of a hard `1[t_A > t_B]` indicator —
this is the smooth, differentiable analogue and is what makes SVI possible.
The probit link is exactly what falls out of the original TrueSkill derivation
once you marginalize the Gaussian outcome-noise variable.

Inference: SVI with a mean-field Normal guide over the player-role skills.
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

ROLES = ["top", "jng", "mid", "adc", "sup"]
N_ROLES = len(ROLES)


def model(matches, n_players, observe_kda=True):
    """
    matches: list of dicts, each with keys:
        team_a:  list of 5 (player_id, role_idx)
        team_b:  list of 5 (player_id, role_idx)
        winner:  1 if team A won, 0 if team B won
        kda_a:   tensor of 5 floats (per-player KDA on team A), or None
        kda_b:   tensor of 5 floats (per-player KDA on team B), or None
    n_players: total number of distinct players in the dataset
    """

    # --- hyperparameters (fixed) ---
    mu_0     = 25.0       # prior skill mean
    sigma_0  = 25.0 / 3   # prior skill std (TrueSkill default)
    beta     = 25.0 / 6   # performance noise
    beta_o   = 1.0        # outcome-link scale
    tau      = 2.0        # KDA observation noise
    alpha    = 0.15       # KDA-per-performance slope
    gamma    = torch.tensor([3.0, 4.0, 4.0, 5.0, 2.0])  # role-specific KDA intercept

    # --- Layer 1: per-role player skill priors ---
    # s has shape (n_players, n_roles); each entry is one latent skill.
    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            s = pyro.sample("s", dist.Normal(mu_0, sigma_0))  # (N_ROLES, n_players)
    s = s.T  # reshape to (n_players, N_ROLES) for easy indexing

    # --- per-match factors ---
    for m_idx, match in enumerate(matches):
        team_a = match["team_a"]  # list of (pid, role_idx)
        team_b = match["team_b"]

        # Layer 2 — performance for each of the 10 players
        # p_i ~ N(s_{pid, role}, beta^2)
        p_a = []
        for slot, (pid, r) in enumerate(team_a):
            p = pyro.sample(f"p_a_{m_idx}_{slot}", dist.Normal(s[pid, r], beta))
            p_a.append(p)
        p_b = []
        for slot, (pid, r) in enumerate(team_b):
            p = pyro.sample(f"p_b_{m_idx}_{slot}", dist.Normal(s[pid, r], beta))
            p_b.append(p)
        p_a = torch.stack(p_a)
        p_b = torch.stack(p_b)

        # Layer 3 — team sums (deterministic factor)
        t_a = p_a.sum()
        t_b = p_b.sum()

        # Layer 4 — outcome factor with probit link
        # P(A wins) = Phi((t_a - t_b) / (sqrt(2) * beta_o))
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample(
            f"y_{m_idx}",
            dist.Bernoulli(win_prob),
            obs=torch.tensor(float(match["winner"])),
        )

        # Layer 5 — per-player KDA observations
        if observe_kda and match.get("kda_a") is not None:
            roles_a = torch.tensor([r for _, r in team_a])
            roles_b = torch.tensor([r for _, r in team_b])
            mean_kda_a = alpha * p_a + gamma[roles_a]
            mean_kda_b = alpha * p_b + gamma[roles_b]
            pyro.sample(
                f"kda_a_{m_idx}",
                dist.Normal(mean_kda_a, tau).to_event(1),
                obs=match["kda_a"],
            )
            pyro.sample(
                f"kda_b_{m_idx}",
                dist.Normal(mean_kda_b, tau).to_event(1),
                obs=match["kda_b"],
            )


def guide(matches, n_players, observe_kda=True):
    """
    Mean-field Gaussian variational posterior over the persistent skill variables.
    The per-match performance variables p_i are also latent, so we need a guide
    for them too — we use a simple Normal guide for each.
    """
    mu_0    = 25.0
    sigma_0 = 25.0 / 3

    # Variational params for player-role skills
    s_loc = pyro.param(
        "s_loc",
        mu_0 * torch.ones(N_ROLES, n_players),
    )
    s_scale = pyro.param(
        "s_scale",
        sigma_0 * torch.ones(N_ROLES, n_players),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            pyro.sample("s", dist.Normal(s_loc, s_scale))

    # Variational params for each per-match performance
    for m_idx, match in enumerate(matches):
        for slot in range(5):
            for side, prefix in [("team_a", "p_a"), ("team_b", "p_b")]:
                name = f"{prefix}_{m_idx}_{slot}"
                loc = pyro.param(f"{name}_loc",   torch.tensor(25.0))
                scl = pyro.param(
                    f"{name}_scale", torch.tensor(4.0),
                    constraint=dist.constraints.positive,
                )
                pyro.sample(name, dist.Normal(loc, scl))


# -----------------------------------------------------------------------------
# Demo / smoke test on synthetic data
# -----------------------------------------------------------------------------
def make_synthetic_data(n_players=20, n_matches=200, seed=0):
    """Generate a tiny synthetic dataset where 'true' skills exist."""
    torch.manual_seed(seed)
    true_s = 25.0 + 5.0 * torch.randn(n_players, N_ROLES)

    matches = []
    for _ in range(n_matches):
        ids = torch.randperm(n_players)[:10].tolist()
        team_a = [(ids[i], i)     for i in range(5)]   # role i for slot i
        team_b = [(ids[5 + i], i) for i in range(5)]

        p_a = torch.tensor([true_s[pid, r] + 4.0 * torch.randn(1).item()
                            for pid, r in team_a])
        p_b = torch.tensor([true_s[pid, r] + 4.0 * torch.randn(1).item()
                            for pid, r in team_b])
        winner = int(p_a.sum() > p_b.sum())

        kda_a = 0.15 * p_a + torch.tensor([3., 4., 4., 5., 2.]) + 2.0 * torch.randn(5)
        kda_b = 0.15 * p_b + torch.tensor([3., 4., 4., 5., 2.]) + 2.0 * torch.randn(5)

        matches.append(dict(
            team_a=team_a, team_b=team_b, winner=winner,
            kda_a=kda_a, kda_b=kda_b,
        ))
    return matches, true_s


def train(matches, n_players, n_steps=2000, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    for step in range(n_steps):
        loss = svi.step(matches, n_players)
        if step % 200 == 0:
            print(f"step {step:5d}   ELBO loss = {loss:,.1f}")
    return pyro.get_param_store()


if __name__ == "__main__":
    n_players = 20
    matches, true_s = make_synthetic_data(n_players=n_players, n_matches=200)
    params = train(matches, n_players, n_steps=2000)

    learned_mu = params["s_loc"].detach().T   # (n_players, n_roles)
    print("\ncorrelation between true and inferred skill (per role):")
    for r, role in enumerate(ROLES):
        corr = torch.corrcoef(
            torch.stack([true_s[:, r], learned_mu[:, r]])
        )[0, 1].item()
        print(f"  {role:>3s}:  r = {corr:+.3f}")