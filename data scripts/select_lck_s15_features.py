from analyze_lck_s15_data import (
    DATA_PATH,
    MODEL_READY_PATH,
    MODEL_READY_ZSCORED_PATH,
    OUT_DIR,
    PLOTS_DIR,
    ROLE_NORMALIZATION_PATH,
    build_model_ready_table,
    build_model_ready_role_zscore_table,
    build_role_normalization_comparison_table,
    load_features,
    plot_model_ready_feature_heatmap,
    plot_role_normalization_comparison,
    write_feature_selection_summary,
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features(DATA_PATH)
    model_ready = build_model_ready_table(features)
    role_comparison = build_role_normalization_comparison_table(model_ready)
    model_ready_zscored = build_model_ready_role_zscore_table(model_ready)

    model_ready.to_csv(MODEL_READY_PATH, index=False)
    role_comparison.to_csv(ROLE_NORMALIZATION_PATH, index=False)
    model_ready_zscored.to_csv(MODEL_READY_ZSCORED_PATH, index=False)
    plot_model_ready_feature_heatmap(
        model_ready, PLOTS_DIR / "model_ready_feature_correlation_heatmap.png"
    )
    plot_role_normalization_comparison(
        role_comparison, PLOTS_DIR / "role_normalization_comparison.png"
    )
    print(f"Saved model-ready file to {MODEL_READY_PATH}")
    print(f"Saved role-normalization comparison file to {ROLE_NORMALIZATION_PATH}")
    print(f"Saved role-zscored model-ready file to {MODEL_READY_ZSCORED_PATH}")
    print(f"Saved feature-selection outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
