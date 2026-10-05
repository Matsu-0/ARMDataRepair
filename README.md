# ARM Data Repair

Anomaly detection and error repair for **multivariate time series**, combining auto-regressive prediction with **domain constraints** (k-NN over a clean reference set).

## Overview

- **Anomaly detection**: Flags points that conflict with predictions and domain constraints.
- **Data repair**: Replaces anomalies with the nearest consistent candidate from the domain-constraint set (guided by the predictor).
- **Pluggable predictors**: VAR (default, pure Java), DLinear / PatchTST (Python).

## Features

- **ARMRepair**: Core algorithm — multi-round forward repair with a configurable prediction model.
- **Baselines**: ERRepair, SCREEN, Lsgreedy, IMR, MTCSC, DomainNN, ModelOnly, CSDI (detect-then-impute).
- **Metrics**: RMSE, Precision, Recall, Time, Regression Loss; forecast RMSE on dirty vs repaired series.
- **Datasets**: Engine, GPS, Road, Weather, Trajectory under `data/`.

## Directory Structure

```
ARMDataRepair/
├── src/main/java/
│   ├── Experiment.java          # Experiment entry and evaluation
│   ├── LoadData.java            # Time series + domain-constraint loading
│   ├── Analysis.java            # Metrics
│   ├── AddNoise.java / LabelData.java
│   └── Algorithm/
│       ├── ARMRepair.java     # Core repair algorithm
│       ├── AnomalyDetector.java
│       ├── EditingRuleRepair.java, SCREEN.java, Lsgreedy.java, IMR.java, MTCSC.java
│       └── util/                # VARUtil, DLinearUtil, PatchTSTUtil, KDTreeUtil, ...
├── python/                      # DLinear, PatchTST, CSDI adapter
│   ├── dlinear_model.py
│   ├── patchtst_model.py
│   ├── csdi_repair.py
│   └── requirements.txt
├── baseline/CSDI/               # Official CSDI sources (NeurIPS 2021)
├── data/                        # Per-domain time series + domain constraints
│   └── <domain>/time_series_data_*.csv, master_data_*.csv
├── model/data/                  # Repaired / prediction series per run
├── results/                     # Logs and ablation CSVs
└── pom.xml
```

> Domain-constraint files are still named `master_data_*.csv` on disk for compatibility with existing datasets.

## Requirements

- **JDK 17** (see `pom.xml`)
- **Maven 3.x**
- **Optional (DLinear / PatchTST)**: Python 3.8+ and dependencies below

### Python dependencies

```bash
pip install -r python/requirements.txt
```

| Package | Version |
|---------|---------|
| numpy   | >=1.21.0, &lt;3.0.0 |
| torch   | >=1.12.0 |

CPU-only PyTorch:

```bash
pip install "numpy>=1.21.0,<3.0.0"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Build

```bash
mvn clean install -U
```

## Run Experiments

Entry point: `Experiment.main`.

Configure paths and which experiment to run in `Experiment.java`:

- **Paths**: `DATA_BASE_PATH` (default `./data/`), `OUTPUT_BASE_PATH` (default `./model/data/`).
- **Uncomment one method** in `main()`:

| Method | Description |
|--------|-------------|
| `varyingModel(5)` | ARM + VAR forecast eval (80/20 split), RMSE/Time. |
| `varyingModelDLinear(5)` | Same with **DLinear** (requires Python). |
| `varyingModelPatchTST(5)` | Same with **PatchTST** (requires Python). |
| `varyingForecastContext(er, idx)` | Ablation: forecast context length (VAR). |
| `varyingForecastHorizon(er, idx)` | Ablation: forecast horizon (VAR). |
| `varyingForecastContextDLinear(...)` | Context ablation with DLinear. |
| `varyingForecastHorizonDLinear(...)` | Horizon ablation with DLinear. |
| `get_arm_repaired(rate)` | ARM-only repair RMSE/Time across datasets. |
| `get_data_repaired(rate)` | ARM vs baselines comparison. |
| `get_csdi_repaired(rate)` | ARM vs CSDI (self-detect then impute) on a 4000-point prefix. |
| `main_td_scale()` | Varying time series length. |
| `main_error_rate()` | Varying error rate. |
| `main_error_range()` | Varying error magnitude. |
| `main_error_length()` | Varying contiguous error length. |
| `main_parameters(...)` | Sweep ARM parameters `k` / `p` / `eta` / `beta`. |
| `whole_data_set(idx)` | Full-series baseline comparison. |

Results go under `results/` (e.g. `expRMSE.txt`, `forecast_horizon/`, `whole_performance/`). Repaired series are written under `model/data/<dataset>/`.

## Data Format

- **Time series**: CSV, one row per timestamp; first column time, remaining columns are variables.
- **Domain constraints**: CSV with the same variable columns (no time column). Each row is a clean reference tuple. Files are named `master_data_*.csv`.

Example (trajectory):

- `data/traj/time_series_data_25168.csv`
- `data/traj/master_data_1180.csv`

Dataset indices in `Experiment.init(dataset_idx)`: `0` engine, `1` gps, `2` road, `3` weather, `4` traj.

## Prediction Models

- **VAR** (default): Pure Java (`VARUtil`). Used with domain-constraint k-NN for repair.
- **DLinear**: `python/dlinear_model.py` via `DLinearUtil`.
- **PatchTST**: `python/patchtst_model.py` via `PatchTSTUtil`.
- **CSDI**: Official code under `baseline/CSDI`; ARM wrapper `python/csdi_repair.py` via `CSDIRepair`. CSDI trains on the dirty series, flags high reconstruction residual, then imputes those points (does not use ARM's error set E). Extra packages: `tqdm` (and the original CSDI `requirements.txt` if you run their scripts).

## License

See the repository root for license information.
