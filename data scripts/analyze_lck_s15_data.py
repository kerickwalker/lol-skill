from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data/lck_s15_games_blocked.csv")
OUT_DIR = Path("data/analysis")
PLOTS_DIR = OUT_DIR / "plots"


def parse_duration_minutes(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) + int(seconds) / 60.0


def zscore_by_role(series: pd.Series, roles: pd.Series) -> pd.Series:
    grouped = series.groupby(roles)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    return ((series - means) / stds).fillna(0.0)


def parse_kda_series(series: pd.Series) -> pd.DataFrame:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace('^="', "", regex=True)
        .str.replace('"$', "", regex=True)
    )
    parts = cleaned.str.split("/", expand=True)
    parts.columns = ["kills", "deaths", "assists"]
    return parts.astype(float)


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    df[["kills", "deaths", "assists"]] = parse_kda_series(df["KDA"])

    df["kp_pct"] = df["KP%"].astype(str).str.rstrip("%").astype(float)
    df["duration_min"] = df["Duration"].astype(str).map(parse_duration_minutes)
    df["win"] = (df["Result"] == "Victory").astype(int)
    df["team_id"] = df["game_block_id"].astype(str) + "_" + df["Result"].astype(str)
    # Smooth the KDA transform so 0-death games are rewarded without creating infinities.
    df["kda_index"] = np.log1p(df["kills"] + df["assists"]) - np.log1p(df["deaths"])
    df["kills_per_min"] = df["kills"] / df["duration_min"]
    df["deaths_per_min"] = df["deaths"] / df["duration_min"]
    df["assists_per_min"] = df["assists"] / df["duration_min"]
    df["dpm_per_csm"] = df["DPM"] / df["CSM"].replace(0, np.nan)

    df["dpm_role_z"] = zscore_by_role(df["DPM"], df["role"])
    df["csm_role_z"] = zscore_by_role(df["CSM"], df["role"])
    df["kp_role_z"] = zscore_by_role(df["kp_pct"], df["role"])
    df["kda_role_z"] = zscore_by_role(df["kda_index"], df["role"])
    df["deaths_role_z"] = zscore_by_role(df["deaths"], df["role"])
    df["performance_score"] = (
        0.35 * df["dpm_role_z"]
        + 0.20 * df["csm_role_z"]
        + 0.20 * df["kp_role_z"]
        + 0.20 * df["kda_role_z"]
        - 0.15 * df["deaths_role_z"]
        + 0.10 * (2 * df["win"] - 1)
    )

    team_totals = (
        df.groupby("team_id")
        .agg(
            team_kills=("kills", "sum"),
            team_deaths=("deaths", "sum"),
            team_assists=("assists", "sum"),
            team_dpm=("DPM", "sum"),
            team_csm=("CSM", "sum"),
            team_mean_perf=("performance_score", "mean"),
        )
        .reset_index()
    )
    df = df.merge(team_totals, on="team_id", how="left")

    df["kill_share"] = df["kills"] / df["team_kills"].replace(0, np.nan)
    df["death_share"] = df["deaths"] / df["team_deaths"].replace(0, np.nan)
    df["assist_share"] = df["assists"] / df["team_assists"].replace(0, np.nan)
    df["dpm_share"] = df["DPM"] / df["team_dpm"].replace(0, np.nan)
    df["csm_share"] = df["CSM"] / df["team_csm"].replace(0, np.nan)

    df["teammate_mean_dpm"] = (df["team_dpm"] - df["DPM"]) / 4.0
    df["teammate_mean_csm"] = (df["team_csm"] - df["CSM"]) / 4.0
    df["teammate_mean_perf"] = ((df["team_mean_perf"] * 5.0) - df["performance_score"]) / 4.0

    opponent_lookup = df[
        [
            "game_block_id",
            "Result",
            "role",
            "player_id",
            "player_name",
            "Champion",
            "DPM",
            "CSM",
            "kp_pct",
            "kills",
            "deaths",
            "assists",
            "performance_score",
        ]
    ].copy()
    opponent_lookup["Result"] = opponent_lookup["Result"].map(
        {"Victory": "Defeat", "Defeat": "Victory"}
    )
    opponent_lookup = opponent_lookup.rename(
        columns={
            "player_id": "opponent_player_id",
            "player_name": "opponent_player_name",
            "Champion": "opponent_champion",
            "DPM": "opponent_dpm",
            "CSM": "opponent_csm",
            "kp_pct": "opponent_kp_pct",
            "kills": "opponent_kills",
            "deaths": "opponent_deaths",
            "assists": "opponent_assists",
            "performance_score": "opponent_performance_score",
        }
    )
    df = df.merge(opponent_lookup, on=["game_block_id", "Result", "role"], how="left")
    df["dpm_diff_vs_role_opp"] = df["DPM"] - df["opponent_dpm"]
    df["csm_diff_vs_role_opp"] = df["CSM"] - df["opponent_csm"]
    df["kp_diff_vs_role_opp"] = df["kp_pct"] - df["opponent_kp_pct"]
    df["perf_diff_vs_role_opp"] = df["performance_score"] - df["opponent_performance_score"]

    return df.sort_values(["game_block_id", "Result", "role", "player_name"]).reset_index(drop=True)


def build_player_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby(["player_id", "player_name", "role"])
        .agg(
            games=("game_block_id", "size"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
            avg_performance=("performance_score", "mean"),
            avg_dpm=("DPM", "mean"),
            avg_csm=("CSM", "mean"),
            avg_kp_pct=("kp_pct", "mean"),
            avg_kda_index=("kda_index", "mean"),
            avg_kills=("kills", "mean"),
            avg_deaths=("deaths", "mean"),
            avg_assists=("assists", "mean"),
            avg_kill_share=("kill_share", "mean"),
            avg_dpm_share=("dpm_share", "mean"),
            avg_teammate_perf=("teammate_mean_perf", "mean"),
            avg_perf_diff_vs_role_opp=("perf_diff_vs_role_opp", "mean"),
        )
        .reset_index()
    )
    summary["wins"] = summary["wins"].astype(int)
    return summary.sort_values(["avg_performance", "games"], ascending=[False, False]).reset_index(drop=True)


def build_pair_summary(features: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        features.groupby(["player_id", "player_name"])["performance_score"]
        .mean()
        .rename("focal_baseline_perf")
        .reset_index()
    )

    pair_rows: list[dict[str, float | int | str]] = []
    for _, team in features.groupby("team_id"):
        records = team.to_dict("records")
        for focal, teammate in permutations(records, 2):
            pair_rows.append(
                {
                    "team_id": focal["team_id"],
                    "game_block_id": focal["game_block_id"],
                    "focal_player_id": focal["player_id"],
                    "focal_player_name": focal["player_name"],
                    "focal_role": focal["role"],
                    "teammate_player_id": teammate["player_id"],
                    "teammate_player_name": teammate["player_name"],
                    "teammate_role": teammate["role"],
                    "win": focal["win"],
                    "focal_performance": focal["performance_score"],
                    "teammate_performance": teammate["performance_score"],
                    "focal_dpm": focal["DPM"],
                    "teammate_dpm": teammate["DPM"],
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs

    summary = (
        pairs.groupby(
            [
                "focal_player_id",
                "focal_player_name",
                "focal_role",
                "teammate_player_id",
                "teammate_player_name",
                "teammate_role",
            ]
        )
        .agg(
            games_together=("game_block_id", "size"),
            win_rate_together=("win", "mean"),
            focal_perf_with_teammate=("focal_performance", "mean"),
            teammate_perf_when_together=("teammate_performance", "mean"),
            focal_dpm_with_teammate=("focal_dpm", "mean"),
            teammate_dpm_when_together=("teammate_dpm", "mean"),
        )
        .reset_index()
    )
    summary = summary.merge(
        baseline,
        left_on=["focal_player_id", "focal_player_name"],
        right_on=["player_id", "player_name"],
        how="left",
    ).drop(columns=["player_id", "player_name"])
    summary["focal_perf_delta_vs_baseline"] = (
        summary["focal_perf_with_teammate"] - summary["focal_baseline_perf"]
    )
    return summary.sort_values(
        ["games_together", "focal_perf_delta_vs_baseline"], ascending=[False, False]
    ).reset_index(drop=True)


def plot_role_distributions(features: pd.DataFrame, out_path: Path) -> None:
    roles = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [
        ("DPM", "Damage Per Minute"),
        ("CSM", "CS Per Minute"),
        ("kp_pct", "Kill Participation %"),
        ("performance_score", "Composite Performance Score"),
    ]

    for ax, (column, title) in zip(axes.flat, metrics):
        values = [features.loc[features["role"] == role, column].dropna() for role in roles]
        ax.boxplot(values, tick_labels=roles, patch_artist=True)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Role-Aware Distribution Check", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_heatmap(features: pd.DataFrame, out_path: Path) -> None:
    columns = [
        "kills",
        "deaths",
        "assists",
        "CSM",
        "DPM",
        "kp_pct",
        "win",
        "dpm_share",
        "teammate_mean_perf",
        "perf_diff_vs_role_opp",
        "performance_score",
    ]
    corr = features[columns].corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_title("Engineered Feature Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_player_skill_scatter(player_summary: pd.DataFrame, out_path: Path) -> None:
    role_colors = {
        "TOP": "#2E86AB",
        "JUNGLE": "#4E9F3D",
        "MID": "#C44536",
        "ADC": "#F4A259",
        "SUPPORT": "#7D5BA6",
    }
    top_players = player_summary[player_summary["games"] >= 20].nlargest(15, "avg_performance")

    fig, ax = plt.subplots(figsize=(11, 8))
    for role, group in player_summary.groupby("role"):
        ax.scatter(
            group["avg_dpm"],
            group["avg_kp_pct"],
            s=group["games"] * 4,
            alpha=0.6,
            label=role,
            color=role_colors.get(role),
            edgecolor="black",
            linewidth=0.3,
        )

    for row in top_players.itertuples(index=False):
        ax.annotate(row.player_name, (row.avg_dpm, row.avg_kp_pct), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title("Player Positioning: Damage, Participation, and Sample Size")
    ax.set_xlabel("Average DPM")
    ax.set_ylabel("Average KP%")
    ax.grid(alpha=0.25)
    ax.legend(title="Role")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_players(player_summary: pd.DataFrame, out_path: Path) -> None:
    top_players = player_summary[player_summary["games"] >= 20].nlargest(20, "avg_performance")
    colors = top_players["win_rate"].map(lambda x: "#2E8B57" if x >= 0.5 else "#B23A48")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top_players["player_name"], top_players["avg_performance"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Top Composite Skill Proxies (min 20 games)")
    ax.set_xlabel("Average performance score")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_teammate_heatmap(pair_summary: pd.DataFrame, out_path: Path) -> None:
    filtered = pair_summary[pair_summary["games_together"] >= 15].copy()
    if filtered.empty:
        return

    top_names = (
        filtered.groupby("focal_player_name")["games_together"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    filtered = filtered[
        filtered["focal_player_name"].isin(top_names)
        & filtered["teammate_player_name"].isin(top_names)
    ]
    if filtered.empty:
        return

    pivot = filtered.pivot_table(
        index="focal_player_name",
        columns="teammate_player_name",
        values="focal_perf_delta_vs_baseline",
        aggfunc="mean",
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(pivot.values, cmap="coolwarm", vmin=-0.35, vmax=0.35)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Teammate Influence Heatmap\n(focal performance delta vs own baseline)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(features: pd.DataFrame, player_summary: pd.DataFrame, pair_summary: pd.DataFrame, out_path: Path) -> None:
    top_players = player_summary[player_summary["games"] >= 20].nlargest(10, "avg_performance")[
        ["player_name", "role", "games", "win_rate", "avg_performance", "avg_dpm", "avg_kp_pct"]
    ]
    top_pairs = pair_summary[pair_summary["games_together"] >= 15].nlargest(
        12, "focal_perf_delta_vs_baseline"
    )[
        [
            "focal_player_name",
            "focal_role",
            "teammate_player_name",
            "teammate_role",
            "games_together",
            "win_rate_together",
            "focal_perf_delta_vs_baseline",
        ]
    ]

    def to_pipe_table(frame: pd.DataFrame) -> str:
        display = frame.round(3).astype(object).fillna("")
        columns = list(display.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in display.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)

    lines = [
        "# LCK S15 Games Blocked Analysis",
        "",
        f"- Rows: {len(features)}",
        f"- Game blocks: {features['game_block_id'].nunique()}",
        f"- Players: {features['player_id'].nunique()}",
        f"- Roles: {', '.join(sorted(features['role'].unique()))}",
        "",
        "## Scope",
        "- All engineered columns come from `data/lck_s15_games_blocked.csv` only.",
        "- No external gold, vision, economy, draft, or timeline data is used in this analysis.",
        "- Each derived feature is either a direct transformation of one CSV column or an aggregation within the same `game_block_id`.",
        "",
        "## Raw Columns Used",
        "- `game_block_id`, `Game`, `Duration`, `player_id`, `player_name`, `role`, `Champion`, `Result`, `KDA`, `CSM`, `DPM`, `KP%`",
        "",
        "## Derived Feature Definitions",
        "- `kills`, `deaths`, `assists`: split directly from `KDA = kills/deaths/assists`.",
        "- `kp_pct`: numeric version of `KP%` after removing the percent sign.",
        "- `duration_min`: match duration converted from `MM:SS` to decimal minutes.",
        "- `win`: `1` for `Victory`, `0` for `Defeat`.",
        "- `team_id`: `game_block_id` combined with `Result`, so each game has one winning team row-group and one losing team row-group.",
        "- `kills_per_min`, `deaths_per_min`, `assists_per_min`: per-minute rates from the parsed KDA counts and `duration_min`.",
        "- `kda_index`: `log(1 + kills + assists) - log(1 + deaths)`.",
        "- `dpm_per_csm`: `DPM / CSM`. This is only a damage-per-farm proxy; it is not gold-based and should not be interpreted as gold efficiency.",
        "- `*_role_z`: z-score of a metric within the same role only.",
        "- `performance_score`: hand-crafted exploratory score, not a learned skill estimate:",
        "  `0.35*dpm_role_z + 0.20*csm_role_z + 0.20*kp_role_z + 0.20*kda_role_z - 0.15*deaths_role_z + 0.10*(2*win - 1)`",
        "- `team_kills`, `team_deaths`, `team_assists`, `team_dpm`, `team_csm`: sums across the five players with the same `team_id`.",
        "- `kill_share`, `death_share`, `assist_share`, `dpm_share`, `csm_share`: player share of the corresponding team total.",
        "- `teammate_mean_dpm`, `teammate_mean_csm`, `teammate_mean_perf`: average of the other four players on the same team in the same game.",
        "- `opponent_*`: stats of the same-role opponent in the same `game_block_id` on the opposite result side.",
        "- `*_diff_vs_role_opp`: player stat minus same-role opponent stat in that match.",
        "",
        "## Modeling Notes",
        "- `kda_index` uses a log-smoothed form so 0-death games are rewarded without producing infinite values.",
        "- Role normalization is used because raw DPM, CSM, and KP differ structurally by role.",
        "- `performance_score` is best treated as an EDA convenience, not as the target latent skill variable.",
        "",
        "## Top Composite Skill Proxies",
        to_pipe_table(top_players),
        "",
        "## Strongest Positive Pair Contexts",
        to_pipe_table(top_pairs),
        "",
        "## Notes",
        "- `performance_score` is a role-normalized proxy, not a learned latent skill.",
        "- `focal_perf_delta_vs_baseline` compares a player's average performance with a given teammate against that player's overall average.",
        "- Opponent-adjusted columns use the same-role opponent in the same game block as a simple matchup control.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features(DATA_PATH)
    player_summary = build_player_summary(features)
    pair_summary = build_pair_summary(features)

    features.to_csv(OUT_DIR / "lck_s15_model_features.csv", index=False)
    player_summary.to_csv(OUT_DIR / "lck_s15_player_summary.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "lck_s15_teammate_pairs.csv", index=False)

    plot_role_distributions(features, PLOTS_DIR / "role_distributions.png")
    plot_feature_heatmap(features, PLOTS_DIR / "feature_correlation_heatmap.png")
    plot_player_skill_scatter(player_summary, PLOTS_DIR / "player_skill_scatter.png")
    plot_top_players(player_summary, PLOTS_DIR / "top_players_skill_proxy.png")
    plot_teammate_heatmap(pair_summary, PLOTS_DIR / "teammate_influence_heatmap.png")

    write_summary(features, player_summary, pair_summary, OUT_DIR / "analysis_summary.md")

    print(f"Saved feature table to {OUT_DIR / 'lck_s15_model_features.csv'}")
    print(f"Saved player summary to {OUT_DIR / 'lck_s15_player_summary.csv'}")
    print(f"Saved pair summary to {OUT_DIR / 'lck_s15_teammate_pairs.csv'}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
