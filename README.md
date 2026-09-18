# Credit Card Fraud Detection — XGBoost + SMOTE Case Study

A reproducible pipeline for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset: feature engineering on ~590K transactions, two imbalance-handling strategies
(class weighting vs. SMOTE), threshold tuning, and feature-importance interpretation
(gain + SHAP).

## Files

| File | What it is |
|---|---|
| `fraud_detection_report.html` | The full write-up — open in any browser. Charts, tables, and interpretation. |
| `fraud_detection_pipeline.py` | The complete pipeline, staged so each step can run (and re-run) on its own. |

## Requirements

```
pip install pandas numpy pyarrow xgboost imbalanced-learn shap scikit-learn matplotlib
```

Python 3.9+. No GPU needed.

## Data

Download `train_transaction.csv` and `train_identity.csv` from the Kaggle competition
page above and place them in a `data/` folder next to the script:

```
your-project/
├── fraud_detection_pipeline.py
└── data/
    ├── train_transaction.csv
    └── train_identity.csv
```

## Running it

Run every stage in order:

```bash
python fraud_detection_pipeline.py all
```

Or run stages one at a time (useful if you want to inspect intermediate output, or if
you're memory-constrained and want each stage to run as its own fresh process):

```bash
python fraud_detection_pipeline.py build_features   # load, merge, clean, engineer features
python fraud_detection_pipeline.py split             # time-based train/test split
python fraud_detection_pipeline.py baseline          # class-weighted XGBoost
python fraud_detection_pipeline.py smote             # RandomUnderSampler + SMOTE, then XGBoost
python fraud_detection_pipeline.py evaluate          # ROC/PR curves, threshold tuning, confusion matrices
python fraud_detection_pipeline.py importance        # gain importance + SHAP summary
```

Each stage reads/writes to `data/` and `figs/` (created automatically), so later stages
depend on earlier ones having already run at least once.

### Output

- `data/features_clean.parquet` — cleaned, fully-numeric feature matrix
- `data/model_baseline.json`, `data/model_smote.json` — trained XGBoost boosters
- `data/results_baseline.json`, `data/results_smote.json` — ROC-AUC / PR-AUC per model
- `data/threshold_summary.json` — precision/recall/F1 at three candidate thresholds
- `data/top_features_gain.json`, `data/top_features_shap.json` — top-20 feature rankings
- `figs/*.png` — all seven report charts

## Method summary

- **Cleaning**: dropped 12 columns >90% missing, engineered time-of-day/day-of-week and
  amount features, frequency-encoded high-cardinality IDs, label-encoded categoricals,
  median-imputed the rest → 427 features, no missing values.
- **Split**: chronological 80/20 (not random) — trains on the past, tests on the future,
  which is how this would actually be deployed and avoids leakage.
- **Model A — baseline**: XGBoost with `scale_pos_weight` (class weighting), no resampling.
- **Model B — SMOTE**: `RandomUnderSampler` trims the majority class first, then `SMOTE`
  oversamples the minority to a 1:2 ratio, then XGBoost. (Full-scale SMOTE on the raw
  ~472K-row training set was impractical on a memory-constrained machine — see the comment
  in `run_smote()` for the reasoning, and how to remove the cap on a bigger machine.)
- **Threshold tuning**: swept 0–1 on Model B; reports the default (0.5), F1-optimal, and a
  90%-recall business target, each with precision/recall/F1 and confusion matrix.
- **Interpretation**: XGBoost gain importance plus SHAP (`TreeExplainer` on a 3,000-row
  test sample) — both are in the report.

## Headline results

| | ROC-AUC | PR-AUC |
|---|---|---|
| Model A — class-weighted | 0.907 | 0.504 |
| Model B — undersample + SMOTE | 0.898 | 0.503 |

Class weighting slightly edges out SMOTE here — a genuine, common result for tree
ensembles, not a bug. Full discussion, plots, and the feature-importance breakdown are in
`fraud_detection_report.html`.
