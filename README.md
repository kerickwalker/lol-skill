# lol-skill
A TrueSkill-like statistical model for modeling the individual skill of professional League of Legends players, fit using Pyro (SVI).

## Setup

```bash
git clone <repo-url>
cd lol-skill
uv sync
```

## Running

Full training run (~1500 steps):
```bash
uv run python3 model.py
```

Quick smoke test (100 steps):
```bash
uv run python3 model.py -n 100
```

## Output

After training, the model prints per-role and overall skill rankings with each player's inferred mean skill (`mu`), uncertainty (`sigma`), and conservative rating (`mu - 3*sigma`).
