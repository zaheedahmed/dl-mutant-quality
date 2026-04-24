# select_configurations.py
#
# Criteria: Median-based High–High (consistent with RQ3) + hit_rate threshold
#
# What it does:
# 1) Reads all results/quality/quality_*_pre_training.csv
# 2) Canonicalizes (operator, config_str) into portable canonical families
# 3) Computes dataset-specific medians for IQ and EQ (per selection dataset)
# 4) Marks mutants in the High–High quadrant using those medians
# 5) Computes hit_rate per canonical family and selects those above thresholds
# 6) Writes outputs for multiple thresholds (0.20, 0.25, 0.30):
#    - results/selection/selected_canonical_configurations_thr_X.XX.csv
#    - results/selection/canonical_mapping_raw_to_canonical.csv (once)

import os
import glob
import pandas as pd

SELECTION_PREFIXES = ("cleanml", "deepfd", "deeplocalize")
QUALITY_DIR = os.path.join("results", "quality")
OUT_DIR = os.path.join("results", "selection")

# Multiple thresholds for sensitivity analysis
HIT_RATE_THRESHOLDS = [0.20, 0.25, 0.30]
DEFAULT_THRESHOLD = 0.25  # For backwards compatibility


def infer_dataset(bug_id: str) -> str:
    bug_id = str(bug_id).lower()
    for p in SELECTION_PREFIXES + ("defect4ml",):
        if bug_id.startswith(p):
            return p
    return "unknown"


def is_int_token(x: str) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


# -------- bins (portable families) --------
def bin_percent(p: float) -> str:
    if p <= 5:
        return "le_5"
    if p <= 15:
        return "5_15"
    if p <= 30:
        return "15_30"
    if p <= 50:
        return "30_50"
    if p <= 70:
        return "50_70"
    if p <= 90:
        return "70_90"
    return "gt_90"


def bin_epochs(e: int) -> str:
    if e <= 5:
        return "le_5"
    if e <= 10:
        return "6_10"
    if e <= 25:
        return "11_25"
    if e <= 50:
        return "26_50"
    if e <= 100:
        return "51_100"
    if e <= 200:
        return "101_200"
    return "gt_200"


def bin_lr(v: float) -> str:
    # absolute log bins (portable even without original lr)
    if v <= 1e-5:
        return "le_1e-5"
    if v <= 1e-4:
        return "1e-5_1e-4"
    if v <= 1e-3:
        return "1e-4_1e-3"
    if v <= 1e-2:
        return "1e-3_1e-2"
    return "gt_1e-2"


def bin_batch_size(b: int) -> str:
    if b <= 16:
        return "le_16"
    if b <= 32:
        return "17_32"
    if b <= 64:
        return "33_64"
    if b <= 128:
        return "65_128"
    if b <= 256:
        return "129_256"
    if b <= 512:
        return "257_512"
    return "gt_512"


def strip_trailing_layer(cfg: str) -> str:
    toks = str(cfg).split("_")
    if toks and is_int_token(toks[-1]):
        return "_".join(toks[:-1]) if len(toks) > 1 else ""
    return str(cfg)


# -------- canonicalization --------
def canonicalize(operator: str, config_str: str) -> str:
    op = str(operator)
    cfg = str(config_str)

    # boolean toggles
    if op in {"disable_batching", "remove_validation_set"}:
        return "toggle"

    # layer-only configs -> operator-only family
    if op in {"remove_bias", "remove_activation_function", "add_bias", "remove_weights_regularisation"}:
        return "any_layer"

    # dropout: rate_rate_layer -> keep rate (or transition if different)
    if op == "change_dropout_rate":
        toks = cfg.split("_")
        if len(toks) >= 3 and is_int_token(toks[-1]):
            a, b = toks[0], toks[1]
            return a if a == b else f"{a}_to_{b}"
        base = strip_trailing_layer(cfg)
        return base if base else "any"

    # activation/initializer/regulariser-like operators: strip trailing layer index
    if op in {
        "change_activation_function",
        "add_activation_function",
        "change_weights_initialisation",
        "add_weights_regularisation",
        "change_weights_regularisation",
    }:
        base = strip_trailing_layer(cfg)
        return base if base else "any"

    # percent-based data mutations
    if op in {
        "add_noise",
        "unbalance_train_data",
        "make_output_classes_overlap",
        "delete_training_data",
        "change_label",
    }:
        try:
            p = float(cfg)
            return f"pct_{bin_percent(p)}"
        except Exception:
            return "pct_unknown"

    # learning rate: e.g., False_0.00055 (take last token)
    if op == "change_learning_rate":
        toks = cfg.split("_")
        try:
            v = float(toks[-1])
            return f"lr_{bin_lr(v)}"
        except Exception:
            return "lr_unknown"

    # epochs: integer
    if op == "change_epochs":
        try:
            e = int(float(cfg))
            return f"ep_{bin_epochs(e)}"
        except Exception:
            return "ep_unknown"

    # batch size: e.g., 128_128 -> bin first token
    if op == "change_batch_size":
        toks = cfg.split("_")
        try:
            b = int(float(toks[0]))
            return f"bs_{bin_batch_size(b)}"
        except Exception:
            return "bs_unknown"

    # discrete-choice operators: keep exact
    if op in {"change_loss_function", "change_optimisation_function"}:
        return cfg

    # fallback: keep raw
    return cfg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(QUALITY_DIR, "quality_*_pre_training.csv")))
    if not files:
        raise FileNotFoundError(f"No quality CSVs found in: {QUALITY_DIR}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        required = {"bug_id", "operator", "config_str", "IQ", "EQ"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(fp)} missing columns: {missing}")

        df["dataset"] = df["bug_id"].apply(infer_dataset)
        df["canonical_key"] = df.apply(lambda r: canonicalize(r["operator"], r["config_str"]), axis=1)
        df["canonical_config_id"] = df["operator"].astype(str) + "::" + df["canonical_key"].astype(str)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Restrict to selection datasets for selection
    sel_df = all_df[all_df["dataset"].isin(SELECTION_PREFIXES)].copy()
    if sel_df.empty:
        raise ValueError("No rows from selection datasets were found. Check bug_id prefixes.")

    # Dataset-specific medians (consistent with RQ3)
    med = (
        sel_df.groupby("dataset")[["IQ", "EQ"]]
        .median()
        .rename(columns={"IQ": "median_IQ", "EQ": "median_EQ"})
        .reset_index()
    )
    sel_df = sel_df.merge(med, on="dataset", how="left")

    # High–High quadrant using medians
    sel_df["is_high_high"] = (sel_df["IQ"] >= sel_df["median_IQ"]) & (sel_df["EQ"] >= sel_df["median_EQ"])

    # Per-canonical-family hit rate stats
    def nunique_where(group: pd.DataFrame, col: str, cond_col: str) -> int:
        return group.loc[group[cond_col], col].nunique()

    grouped = sel_df.groupby(["operator", "canonical_key", "canonical_config_id"], as_index=False)

    agg = grouped.apply(
        lambda g: pd.Series(
            {
                "total_mutants": len(g),
                "high_high_mutants": int(g["is_high_high"].sum()),
                "hit_rate": float(g["is_high_high"].mean()) if len(g) else 0.0,
                "support_bugs": g["bug_id"].nunique(),
                "support_datasets": g["dataset"].nunique(),
                "support_bugs_high_high": nunique_where(g, "bug_id", "is_high_high"),
                "support_datasets_high_high": nunique_where(g, "dataset", "is_high_high"),
            }
        )
    ).reset_index(drop=True)

    # Save results for each threshold
    for thr in HIT_RATE_THRESHOLDS:
        selected = agg[agg["hit_rate"] >= thr].copy()
        selected = selected.sort_values(
            ["hit_rate", "support_datasets_high_high", "support_bugs_high_high", "high_high_mutants"],
            ascending=[False, False, False, False],
        )

        out_selected = os.path.join(OUT_DIR, f"selected_canonical_configurations_thr_{thr:.2f}.csv")
        selected.to_csv(out_selected, index=False)
        
        print(f"\nThreshold {thr:.2f}:")
        print(f"  Selected canonical configs: {len(selected)}")
        print(f"  Saved to: {out_selected}")

    # Also save with default threshold for backwards compatibility
    default_selected = agg[agg["hit_rate"] >= DEFAULT_THRESHOLD].copy()
    default_selected = default_selected.sort_values(
        ["hit_rate", "support_datasets_high_high", "support_bugs_high_high", "high_high_mutants"],
        ascending=[False, False, False, False],
    )
    out_default = os.path.join(OUT_DIR, "selected_canonical_configurations.csv")
    default_selected.to_csv(out_default, index=False)

    # Full mapping for transparency/debugging (only save once)
    mapping = sel_df[
        [
            "dataset",
            "bug_id",
            "operator",
            "config_str",
            "canonical_key",
            "canonical_config_id",
            "IQ",
            "EQ",
            "median_IQ",
            "median_EQ",
            "is_high_high",
        ]
    ].copy()
    out_map = os.path.join(OUT_DIR, "canonical_mapping_raw_to_canonical.csv")
    mapping.to_csv(out_map, index=False)

    total_canon = sel_df["canonical_config_id"].nunique()

    print("\n=== Canonical Selection Summary (Criteria: medians + hit_rate) ===")
    print(f"Quality files read: {len(files)}")
    print(f"Selection datasets: {SELECTION_PREFIXES}")
    print("\nDataset medians (used for High–High):")
    for _, r in med.iterrows():
        print(f"  {r['dataset']}: median_IQ={r['median_IQ']:.4f}, median_EQ={r['median_EQ']:.4f}")

    print(f"\nUnique canonical_config_id in selection datasets: {total_canon}")
    
    print("\nSelection results by threshold:")
    for thr in HIT_RATE_THRESHOLDS:
        selected_canon = len(agg[agg["hit_rate"] >= thr])
        if total_canon > 0:
            rr = 1.0 - (selected_canon / total_canon)
            print(f"  Threshold {thr:.2f}: {selected_canon} configs selected (reduction ratio: {rr:.4f})")

    print(f"\nSaved mapping: {out_map}")
    print(f"Saved default selection: {out_default}")

if __name__ == "__main__":
    main()