# Quality-Driven Selective Mutation for Deep Learning - Replication Package

## Overview

This replication package enables reproduction of the experimental results reported in our paper. The package includes pre-computed execution matrices and complete analysis scripts to reproduce all quality assessments, configuration selections, and validation results.

## Requirements

**Software/Hardware:**
- Python 3.8
- Standard desktop/laptop

## Scope

**Included:**
- Pre-computed execution matrices (in `results/execution_matrix/`)
- Complete analysis pipeline (killing probability → quality metrics → selection → validation)

**Not included:**
- Trained models and mutants (excluded due to size; execution matrix captures all necessary behavioral information)

## Directory Structure

```
dl-mutant-quality/
├── execution_matrix.py          # [Reference only - already executed]
├── killing_probability.py       # Step 1: Aggregate to probabilities
├── quantify_quality.py          # Step 2: Compute IQ and EQ
├── select_configurations.py     # Step 3: Select configurations
├── validate_configurations.py   # Step 4: Validate on Defect4ML
├── visualize_results.ipynb      # Step 5: Generate figures
├── results/
│   ├── execution_matrix/        # [Provided] Pre-computed data
│   ├── killing_probability/     # [Generated in Step 1]
│   ├── quality/                 # [Generated in Step 2]
│   └── selection/               # [Generated in Steps 3-4]
└── figures/                     # [Generated in Step 5]
```

## Reproduction Steps

### Step 1: Calculate Killing Probabilities
```bash
python killing_probability.py
```
Aggregates multiple training runs into per-test killing probabilities with confidence intervals.

**Output:** `results/killing_probability/killing_probability_<bug>_pre_training.{pkl,csv}`

### Step 2: Compute Quality Metrics
```bash
python quantify_quality.py
```
Computes IQ (intrinsic quality) and EQ (extrinsic quality) for each mutant.

**Output:** `results/quality/quality_<bug>_pre_training.{pkl,csv}`

### Step 3: Select Canonical Configurations
```bash
python select_configurations.py
```
Performs canonicalization and selects configurations based on High-High quadrant hit rates. Generates results for three thresholds (0.20, 0.25, 0.30).

**Output:** 
- `results/selection/selected_canonical_configurations_thr_{0.20,0.25,0.30}.csv`
- `results/selection/canonical_mapping_raw_to_canonical.csv`

### Step 4: Validate on Held-Out Dataset
```bash
python validate_configurations.py
```
Validates selected configurations on Defect4ML dataset.

**Output:** `results/selection/defect4ml_validation_summary.csv`

### Step 5: Generate Visualizations
```bash
jupyter notebook visualize_results.ipynb
```
Generates all figures presented in the paper from the computed results. Run all cells sequentially.

**Output:** Figures saved in `figures/`

## Key Output Files

### Quality Metrics (Step 2)
CSV files with columns: `bug_id`, `operator`, `config_str`, `IQ`, `EQ`, `S_m`, `weighted_C`

### Selected Configurations (Step 3)
CSV files with columns: `canonical_config_id`, `hit_rate`, `total_mutants`, `high_high_mutants`, `support_bugs`, `support_datasets`

### Validation Summary (Step 4)
CSV file with columns: `hit_rate_threshold`, `selected_families`, `defect4ml_reduction_ratio`, `defect4ml_median_IQ/EQ_before/after`, `defect4ml_high_high_prop_before/after`

## Expected Results

- **Reduction ratio:** ~56% of mutants eliminated (threshold = 0.25)
- **Quality preservation:** Median IQ/EQ maintained or improved on defect4ML
- **High-High retention:** Preserved or increased on held-out dataset

## Configuration

Each script has a configuration section at the top:
- `BUG_LIST`: Bugs to process (auto-loaded from execution_matrix.py)
