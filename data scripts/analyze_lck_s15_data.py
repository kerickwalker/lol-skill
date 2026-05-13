from __future__ import annotations

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/lck_s15_games.csv"
OUT_DIR = PROJECT_ROOT / "data/analysis"
PLOTS_DIR = OUT_DIR / "plots"
MODEL_READY_PATH = PROJECT_ROOT / "data/lck_s15_games_MODEL-READY.csv"
ROLE_NORMALIZATION_PATH = PROJECT_ROOT / "data/role_normalization_comparison.csv"


def parse_duration_minutes(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) + int(seconds) / 60.0


def zscore_by_role(series: pd.Series, roles: pd.Series) -> pd.Series:
    grouped = series.groupby(roles)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    return ((series - means) / stds).fillna(0.0)


def center_by_role(series: pd.Series, roles: pd.Series) -> pd.Series:
    means = series.groupby(roles).transform("mean")
    return series - means


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


def build_model_ready_table(features: pd.DataFrame) -> pd.DataFrame:
    model = features.copy()
    recreate_cols = [
        "team_kills",
        "team_deaths",
        "team_assists",
        "team_cs",
        "team_golds",
        "team_vision_score",
        "team_total_damage_to_champion",
        "opponent_kills",
        "opponent_deaths",
        "opponent_assists",
        "opponent_cs",
        "opponent_golds",
        "opponent_vision_score",
        "opponent_total_damage_to_champion",
        "kills_diff_vs_role_opp",
        "deaths_diff_vs_role_opp",
        "assists_diff_vs_role_opp",
        "cs_diff_vs_role_opp",
        "golds_diff_vs_role_opp",
        "vision_diff_vs_role_opp",
        "damage_diff_vs_role_opp",
    ]
    model = model.drop(columns=[col for col in recreate_cols if col in model.columns], errors="ignore")

    base_numeric_cols = [
        "level",
        "kills",
        "deaths",
        "assists",
        "cs",
        "golds",
        "vision_score",
        "solo_kills",
        "double_kills",
        "triple_kills",
        "quadra_kills",
        "penta_kills",
        "gd_at_15",
        "csd_at_15",
        "xpd_at_15",
        "objectives_stolen",
        "damage_dealt_to_buildings",
        "total_heal",
        "total_heals_on_teammates",
        "damage_self_mitigated",
        "total_damage_shielded_on_teammates",
        "total_time_cc_dealt",
        "total_damage_taken",
        "total_time_spent_dead",
        "shutdown_bounty_collected",
        "shutdown_bounty_lost",
        "total_damage_to_champion",
    ]
    for col in base_numeric_cols:
        model[col] = pd.to_numeric(model[col], errors="coerce")

    team_totals = (
        model.groupby("team_id")
        .agg(
            team_kills=("kills", "sum"),
            team_deaths=("deaths", "sum"),
            team_assists=("assists", "sum"),
            team_cs=("cs", "sum"),
            team_golds=("golds", "sum"),
            team_vision_score=("vision_score", "sum"),
            team_total_damage_to_champion=("total_damage_to_champion", "sum"),
        )
        .reset_index()
    )
    model = model.merge(team_totals, on="team_id", how="left")

    opponent_lookup = model[
        [
            "game_block_id",
            "Result",
            "role",
            "kills",
            "deaths",
            "assists",
            "cs",
            "golds",
            "vision_score",
            "total_damage_to_champion",
        ]
    ].copy()
    opponent_lookup["Result"] = opponent_lookup["Result"].map(
        {"Victory": "Defeat", "Defeat": "Victory"}
    )
    opponent_lookup = opponent_lookup.rename(
        columns={
            "kills": "opponent_kills",
            "deaths": "opponent_deaths",
            "assists": "opponent_assists",
            "cs": "opponent_cs",
            "golds": "opponent_golds",
            "vision_score": "opponent_vision_score",
            "total_damage_to_champion": "opponent_total_damage_to_champion",
        }
    )
    model = model.merge(opponent_lookup, on=["game_block_id", "Result", "role"], how="left")
    model["kills_diff_vs_role_opp"] = model["kills"] - model["opponent_kills"]
    model["deaths_diff_vs_role_opp"] = model["deaths"] - model["opponent_deaths"]
    model["assists_diff_vs_role_opp"] = model["assists"] - model["opponent_assists"]
    model["cs_diff_vs_role_opp"] = model["cs"] - model["opponent_cs"]
    model["golds_diff_vs_role_opp"] = model["golds"] - model["opponent_golds"]
    model["vision_diff_vs_role_opp"] = model["vision_score"] - model["opponent_vision_score"]
    model["damage_diff_vs_role_opp"] = (
        model["total_damage_to_champion"] - model["opponent_total_damage_to_champion"]
    )

    ordered_columns = [
        "game_block_id",
        "game_id",
        "Date",
        "Tournament",
        "Game",
        "player_id",
        "player_name",
        "role",
        "Champion",
        "Result",
        "Duration",
        "level",
        "kills",
        "deaths",
        "assists",
        "cs",
        "golds",
        "vision_score",
        "solo_kills",
        "double_kills",
        "triple_kills",
        "quadra_kills",
        "penta_kills",
        "gd_at_15",
        "csd_at_15",
        "xpd_at_15",
        "objectives_stolen",
        "damage_dealt_to_buildings",
        "total_heal",
        "total_heals_on_teammates",
        "damage_self_mitigated",
        "total_damage_shielded_on_teammates",
        "total_time_cc_dealt",
        "total_damage_taken",
        "total_time_spent_dead",
        "shutdown_bounty_collected",
        "shutdown_bounty_lost",
        "total_damage_to_champion",
        "team_kills",
        "team_deaths",
        "team_assists",
        "team_cs",
        "team_golds",
        "team_vision_score",
        "team_total_damage_to_champion",
        "kills_diff_vs_role_opp",
        "deaths_diff_vs_role_opp",
        "assists_diff_vs_role_opp",
        "cs_diff_vs_role_opp",
        "golds_diff_vs_role_opp",
        "vision_diff_vs_role_opp",
        "damage_diff_vs_role_opp",
    ]
    return model[ordered_columns].sort_values(
        ["game_block_id", "Result", "role", "player_name"]
    ).reset_index(drop=True)


def build_role_normalization_comparison_table(model_ready: pd.DataFrame) -> pd.DataFrame:
    comparison = model_ready[
        [
            "game_block_id",
            "game_id",
            "Date",
            "Tournament",
            "Game",
            "player_id",
            "player_name",
            "role",
            "Champion",
            "Result",
            "Duration",
            "kills",
            "cs",
            "golds",
            "vision_score",
            "total_heals_on_teammates",
            "damage_self_mitigated",
            "total_damage_to_champion",
        ]
    ].copy()

    role_normalized_stats = [
        "kills",
        "cs",
        "golds",
        "vision_score",
        "total_heals_on_teammates",
        "damage_self_mitigated",
        "total_damage_to_champion",
    ]
    for stat in role_normalized_stats:
        comparison[f"{stat}_minus_role_mean"] = center_by_role(comparison[stat], comparison["role"])
        comparison[f"{stat}_role_z"] = zscore_by_role(comparison[stat], comparison["role"])

    return comparison.sort_values(["game_block_id", "Result", "role", "player_name"]).reset_index(drop=True)


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


def plot_model_ready_feature_heatmap(model_ready: pd.DataFrame, out_path: Path) -> None:
    corr_frame = model_ready.copy()
    corr_frame["Result"] = (corr_frame["Result"] == "Victory").astype(int)
    corr_frame["Duration"] = corr_frame["Duration"].astype(str).map(parse_duration_minutes)

    columns = list(corr_frame.columns[corr_frame.columns.get_loc("Result") :])
    corr = corr_frame[columns].corr().round(2)

    # Wider figure so the full selected feature set remains readable.
    fig, ax = plt.subplots(figsize=(20, 18))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=8)
    ax.set_title("Model-Ready Feature Correlation Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_role_normalization_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    stats = [
        ("kills", "Kills"),
        ("cs", "CS"),
        ("golds", "Gold"),
        ("vision_score", "Vision Score"),
        ("total_heals_on_teammates", "Heals On Teammates"),
        ("damage_self_mitigated", "Damage Self Mitigated"),
        ("total_damage_to_champion", "Damage To Champions"),
    ]
    roles = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]

    fig, axes = plt.subplots(len(stats), 3, figsize=(18, 3.6 * len(stats)))
    if len(stats) == 1:
        axes = np.array([axes])

    for row_idx, (stat, title) in enumerate(stats):
        raw_ax, centered_ax, z_ax = axes[row_idx]

        raw_values = [comparison.loc[comparison["role"] == role, stat].dropna() for role in roles]
        centered_values = [
            comparison.loc[comparison["role"] == role, f"{stat}_minus_role_mean"].dropna()
            for role in roles
        ]
        z_values = [comparison.loc[comparison["role"] == role, f"{stat}_role_z"].dropna() for role in roles]

        raw_ax.boxplot(raw_values, tick_labels=roles, patch_artist=True)
        centered_ax.boxplot(centered_values, tick_labels=roles, patch_artist=True)
        z_ax.boxplot(z_values, tick_labels=roles, patch_artist=True)

        raw_ax.set_title(f"{title}: Raw")
        centered_ax.set_title(f"{title}: Minus Role Mean")
        z_ax.set_title(f"{title}: Role Z-Score")

        for ax in (raw_ax, centered_ax, z_ax):
            ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Role Normalization Comparison For Selected Model Inputs", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
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


def write_comprehension_summary(
    features: pd.DataFrame, player_summary: pd.DataFrame, pair_summary: pd.DataFrame, out_path: Path
) -> None:
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
        "# LCK S15 Data Comprehension Summary",
        "",
        f"- Rows: {len(features)}",
        f"- Game blocks: {features['game_block_id'].nunique()}",
        f"- Players: {features['player_id'].nunique()}",
        f"- Roles: {', '.join(sorted(features['role'].unique()))}",
        "",
        "## Scope",
        "- All engineered columns come from `data/lck_s15_games.csv` only.",
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


def write_feature_selection_summary(out_path: Path) -> None:
    lines = [
        "# LCK S15 Feature Selection Summary",
        "",
        "## Purpose",
        "- This file records the current model-ready feature choices.",
        "- It is meant to stay separate from the general comprehension/EDA notes.",
        "",
        "## Current Model-Ready Export",
        "- `data/lck_s15_games_MODEL-READY.csv` is produced by `data scripts/analyze_lck_s15_data.py` from `data/lck_s15_games.csv`.",
        "- One row per player-game.",
        "- `Duration` is kept as an explicit raw input rather than replacing it with per-minute derived stats.",
        "- This file is intentionally kept raw for now.",
        "",
        "## Included Raw Player Stats",
        "- `Result`, `Duration`, `level`, `kills`, `deaths`, `assists`, `cs`, `golds`, `vision_score`",
        "- `solo_kills`, `double_kills`, `triple_kills`, `quadra_kills`, `penta_kills`",
        "- `gd_at_15`, `csd_at_15`, `xpd_at_15`",
        "- `objectives_stolen`, `damage_dealt_to_buildings`",
        "- `total_heal`, `total_heals_on_teammates`, `damage_self_mitigated`, `total_damage_shielded_on_teammates`",
        "- `total_time_cc_dealt`, `total_damage_taken`, `total_time_spent_dead`",
        "- `shutdown_bounty_collected`, `shutdown_bounty_lost`, `total_damage_to_champion`",
        "",
        "## Excluded Raw / Derived Stats",
        "- `KDA`",
        "- `CSM`, `DPM`, `gpm` and similar per-minute rate features in the model-ready export",
        "- `consumables_purchased`, `items_purchased`",
        "- ward subcomponent fields such as `wards_placed`, `wards_destroyed`, `control_wards_purchased`",
        "- `damage_dealt_to_turrets`, `time_ccing_others`",
        "",
        "## Included Team Context",
        "- `team_kills`, `team_deaths`, `team_assists`, `team_cs`, `team_golds`, `team_vision_score`, `team_total_damage_to_champion`",
        "- These are included explicitly so the model can see team-level context without needing the full 10-player game table at inference time.",
        "",
        "## Included Same-Role Opponent Differences",
        "- `kills_diff_vs_role_opp`, `deaths_diff_vs_role_opp`, `assists_diff_vs_role_opp`, `cs_diff_vs_role_opp`",
        "- `golds_diff_vs_role_opp`, `vision_diff_vs_role_opp`, `damage_diff_vs_role_opp`",
        "",
        "## Role-Normalization Comparison File",
        "- `data/role_normalization_comparison.csv` is a separate comparison file and is not meant to be mixed into the training table by default.",
        "- For `kills`, `cs`, `golds`, `vision_score`, `total_heals_on_teammates`, `damage_self_mitigated`, and `total_damage_to_champion`, it includes:",
        "- `*_minus_role_mean`",
        "- `*_role_z`",
        "- These are included for comparison and discussion, not yet as a final commitment to one normalization strategy.",
        "",
        "## Analysis Plots Relevant To Selection",
        "- `plots/model_ready_feature_correlation_heatmap.png`: correlation matrix over all columns from `Result` onward in the model-ready CSV, with `Result` encoded as `1/0` and `Duration` converted to decimal minutes.",
        "- `plots/role_normalization_comparison.png`: compares raw, role-centered, and role-z-scored versions of the selected normalization stats.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features(DATA_PATH)
    model_ready = build_model_ready_table(features)
    role_comparison = build_role_normalization_comparison_table(model_ready)
    player_summary = build_player_summary(features)
    pair_summary = build_pair_summary(features)

    features.to_csv(OUT_DIR / "lck_s15_model_features.csv", index=False)
    player_summary.to_csv(OUT_DIR / "lck_s15_player_summary.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "lck_s15_teammate_pairs.csv", index=False)
    model_ready.to_csv(MODEL_READY_PATH, index=False)

    plot_role_distributions(features, PLOTS_DIR / "role_distributions.png")
    plot_feature_heatmap(features, PLOTS_DIR / "feature_correlation_heatmap.png")
    plot_model_ready_feature_heatmap(model_ready, PLOTS_DIR / "model_ready_feature_correlation_heatmap.png")
    plot_role_normalization_comparison(
        role_comparison, PLOTS_DIR / "role_normalization_comparison.png"
    )
    plot_player_skill_scatter(player_summary, PLOTS_DIR / "player_skill_scatter.png")
    plot_top_players(player_summary, PLOTS_DIR / "top_players_skill_proxy.png")
    plot_teammate_heatmap(pair_summary, PLOTS_DIR / "teammate_influence_heatmap.png")

    comprehension_path = OUT_DIR / "_analysis_comprehension.tmp.md"
    selection_path = OUT_DIR / "_feature_selection_summary.tmp.md"
    summary_path = OUT_DIR / "analysis.md"
    write_comprehension_summary(features, player_summary, pair_summary, comprehension_path)
    write_feature_selection_summary(selection_path)
    combined_summary = (
        comprehension_path.read_text(encoding="utf-8").rstrip()
        + "\n\n---\n\n"
        + selection_path.read_text(encoding="utf-8").lstrip()
    )
    summary_path.write_text(combined_summary, encoding="utf-8")
    comprehension_path.unlink(missing_ok=True)
    selection_path.unlink(missing_ok=True)
    (OUT_DIR / "analysis_summary.md").unlink(missing_ok=True)
    (OUT_DIR / "analysis_comprehension.md").unlink(missing_ok=True)
    (OUT_DIR / "feature_selection_summary.md").unlink(missing_ok=True)

    role_comparison.to_csv(ROLE_NORMALIZATION_PATH, index=False)

    print(f"Saved feature table to {OUT_DIR / 'lck_s15_model_features.csv'}")
    print(f"Saved model-ready file to {MODEL_READY_PATH}")
    print(f"Saved role-normalization comparison file to {ROLE_NORMALIZATION_PATH}")
    print(f"Saved player summary to {OUT_DIR / 'lck_s15_player_summary.csv'}")
    print(f"Saved pair summary to {OUT_DIR / 'lck_s15_teammate_pairs.csv'}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
