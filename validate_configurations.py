# validate_configurations.py
#
# Validates the selected canonical configurations on held-out defect4ML dataset
#
# What it does:
# 1) Reads the selected configurations from the selection script outputs
# 2) Loads all quality data and applies the same canonicalization
# 3) Filters defect4ML mutants by selected canonical families
# 4) Computes validation metrics (reduction ratio, quality preservation)
# 5) Writes summary: results/selection/defect4ml_validation_summary.csv

import os
import glob
import pandas as pd

QUALITY_DIR = os.path.join("results", "quality")
SELECTION_DIR = os.path.join("results", "selection")
OUT_DIR = os.path.join("results", "selection")

SELECTION_PREFIXES = ("cleanml", "deepfd", "deeplocalize")
DEFECT_PREFIX = "defect4ml"

THRESHOLDS = [0.20, 0.25, 0.30]


def infer_dataset(bug_id: str) -> str:
    s = str(bug_id).lower()
    for p in SELECTION_PREFIXES + (DEFECT_PREFIX,):
        if s.startswith(p):
            return p
    return "unknown"


def is_int_token(x: str) -> bool:
    try:
        int(x)
        return True
    except Exception:
        return False


def bin_percent(p: float) -> str:
    if p <= 5: return "le_5"
    if p <= 15: return "5_15"
    if p <= 30: return "15_30"
    if p <= 50: return "30_50"
    if p <= 70: return "50_70"
    if p <= 90: return "70_90"
    return "gt_90"


def bin_epochs(e: int) -> str:
    if e <= 5: return "le_5"
    if e <= 10: return "6_10"
    if e <= 25: return "11_25"
    if e <= 50: return "26_50"
    if e <= 100: return "51_100"
    if e <= 200: return "101_200"
    return "gt_200"


def bin_lr(v: float) -> str:
    if v <= 1e-5: return "le_1e-5"
    if v <= 1e-4: return "1e-5_1e-4"
    if v <= 1e-3: return "1e-4_1e-3"
    if v <= 1e-2: return "1e-3_1e-2"
    return "gt_1e-2"


def bin_batch_size(b: int) -> str:
    if b <= 16: return "le_16"
    if b <= 32: return "17_32"
    if b <= 64: return "33_64"
    if b <= 128: return "65_128"
    if b <= 256: return "129_256"
    if b <= 512: return "257_512"
    return "gt_512"


def strip_trailing_layer(cfg: str) -> str:
    toks = str(cfg).split("_")
    if toks and is_int_token(toks[-1]):
        return "_".join(toks[:-1]) if len(toks) > 1 else ""
    return str(cfg)


def canonicalize(operator: str, config_str: str) -> str:
    """Same canonicalization function as in selection script"""
    op = str(operator)
    cfg = str(config_str)

    if op in {"disable_batching", "remove_validation_set"}:
        return "toggle"

    if op in {"remove_bias", "remove_activation_function", "add_bias", "remove_weights_regularisation"}:
        return "any_layer"

    if op == "change_dropout_rate":
        toks = cfg.split("_")
        if len(toks) >= 3 and is_int_token(toks[-1]):
            a, b = toks[0], toks[1]
            return a if a == b else f"{a}_to_{b}"
        base = strip_trailing_layer(cfg)
        return base if base else "any"

    if op in {
        "change_activation_function",
        "add_activation_function",
        "change_weights_initialisation",
        "add_weights_regularisation",
        "change_weights_regularisation",
    }:
        base = strip_trailing_layer(cfg)
        return base if base else "any"

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

    if op == "change_learning_rate":
        toks = cfg.split("_")
        try:
            v = float(toks[-1])
            return f"lr_{bin_lr(v)}"
        except Exception:
            return "lr_unknown"

    if op == "change_epochs":
        try:
            e = int(float(cfg))
            return f"ep_{bin_epochs(e)}"
        except Exception:
            return "ep_unknown"

    if op == "change_batch_size":
        toks = cfg.split("_")
        try:
            b = int(float(toks[0]))
            return f"bs_{bin_batch_size(b)}"
        except Exception:
            return "bs_unknown"

    if op in {"change_loss_function", "change_optimisation_function"}:
        return cfg

    return cfg


def load_selected_configs(threshold: float) -> set:
    """Load selected canonical configuration IDs for a given threshold"""
    filename = os.path.join(SELECTION_DIR, f"selected_canonical_configurations_thr_{threshold:.2f}.csv")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Selection file not found: {filename}. Run selection script first.")
    
    df = pd.read_csv(filename)
    if "canonical_config_id" not in df.columns:
        raise ValueError(f"Missing 'canonical_config_id' column in {filename}")
    
    return set(df["canonical_config_id"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all quality data
    files = sorted(glob.glob(os.path.join(QUALITY_DIR, "quality_*_pre_training.csv")))
    if not files:
        raise FileNotFoundError(f"No quality CSVs found in {QUALITY_DIR}")

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

    # Separate Defect4ML data for validation
    def_df = all_df[all_df["dataset"] == DEFECT_PREFIX].copy()
    
    if def_df.empty:
        raise ValueError("No defect4ML rows found. Ensure bug_id starts with 'defect4ml'.")

    # Baseline metrics for Defect4ML
    base_n = len(def_df)
    base_med_iq = float(def_df["IQ"].median())
    base_med_eq = float(def_df["EQ"].median())
    
    # High-High calculation for Defect4ML (using its own medians)
    def_df["is_high_high_def"] = (def_df["IQ"] >= base_med_iq) & (def_df["EQ"] >= base_med_eq)
    base_hh_prop = float(def_df["is_high_high_def"].mean())

    # Validate for each threshold
    rows = []
    for thr in THRESHOLDS:
        try:
            # Load selected configurations from file
            selected_ids = load_selected_configs(thr)
            
            # Filter Defect4ML by selected canonical families
            def_keep = def_df[def_df["canonical_config_id"].isin(selected_ids)].copy()

            kept_n = len(def_keep)
            rr_mutants = 1.0 - (kept_n / base_n) if base_n > 0 else 0.0

            kept_med_iq = float(def_keep["IQ"].median()) if kept_n > 0 else float("nan")
            kept_med_eq = float(def_keep["EQ"].median()) if kept_n > 0 else float("nan")

            kept_hh_prop = float(def_keep["is_high_high_def"].mean()) if kept_n > 0 else float("nan")

            rows.append({
                "hit_rate_threshold": thr,
                "selected_families": len(selected_ids),
                "defect4ml_mutants_before": base_n,
                "defect4ml_mutants_after": kept_n,
                "defect4ml_reduction_ratio": rr_mutants,
                "defect4ml_median_IQ_before": base_med_iq,
                "defect4ml_median_IQ_after": kept_med_iq,
                "defect4ml_median_EQ_before": base_med_eq,
                "defect4ml_median_EQ_after": kept_med_eq,
                "defect4ml_high_high_prop_before": base_hh_prop,
                "defect4ml_high_high_prop_after": kept_hh_prop,
            })
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    if not rows:
        raise RuntimeError("No validation results generated. Ensure selection script has been run.")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "defect4ml_validation_summary.csv")
    out.to_csv(out_path, index=False)

    print("=== defect4ML Validation (Criteria: medians + hit_rate) ===")
    print(f"Defect4ML mutants (baseline): {base_n}")
    print(f"Baseline medians: IQ={base_med_iq:.4f}, EQ={base_med_eq:.4f}")
    print(f"Baseline High–High proportion: {base_hh_prop:.4f}")
    print()

    for _, r in out.iterrows():
        print(
            f"thr={r['hit_rate_threshold']:.2f} | families={int(r['selected_families'])} | "
            f"mutants {int(r['defect4ml_mutants_before'])}->{int(r['defect4ml_mutants_after'])} "
            f"(RR={r['defect4ml_reduction_ratio']:.4f}) | "
            f"median IQ {r['defect4ml_median_IQ_before']:.4f}->{r['defect4ml_median_IQ_after']:.4f} | "
            f"median EQ {r['defect4ml_median_EQ_before']:.4f}->{r['defect4ml_median_EQ_after']:.4f} | "
            f"High–High {r['defect4ml_high_high_prop_before']:.4f}->{r['defect4ml_high_high_prop_after']:.4f}"
        )

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()