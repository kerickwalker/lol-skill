from pathlib import Path

from analyze_lck_s15_data import (
    DATA_PATH,
    OUT_DIR,
    PLOTS_DIR,
    build_pair_summary,
    build_player_summary,
    load_features,
    plot_feature_heatmap,
    plot_player_skill_scatter,
    plot_role_distributions,
    plot_teammate_heatmap,
    plot_top_players,
)


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

    print(f"Saved comprehension analysis tables to {OUT_DIR}")
    print(f"Saved comprehension plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
