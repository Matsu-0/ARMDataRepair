# ARM: Auto-Regression Model and Domain Constraints Data Repairer

Anomaly detection and error repair for **multivariate time series** using master data. The method combines auto-regressive prediction with k-NN search over master data to detect and repair anomalies.

## Overview

- **Anomaly detection**: Identifies anomalous points by comparing time series with predictions and master data.
- **Data repair**: Repairs dirty time series by replacing anomalies with the nearest consistent value from master data (guided by prediction).
- **Pluggable prediction models**: VAR (default), DLinear (Python), and PatchTST (Python) for the auto-regressive component.

## Features

- **MasterDetector**: Core algorithm—forward/backward repair with configurable prediction model.
- **Multiple baselines**: ERRepair, SCREEN, Lsgreedy, IMR, MTCSC for comparison.
- **Metrics**: RMSE, Precision, Recall, Time, Regression Loss (raw and normalized).
- **Datasets**: Engine, GPS, Road, Weather, Trajectory (included under `data/`).

## Directory Structure

```
MasterDetector/
├── src/main/java/          # Java source
│   ├── Experiment.java     # Experiment entry and evaluation
│   ├── LoadData.java      # Data loading
│   ├── Analysis.java      # Metrics (RMSE, Precision, Recall)
│   ├── AddNoise.java      # Synthetic error injection
│   ├── Algorithm/         # Core algorithms
│   │   ├── MasterDetector.java
│   │   ├── AnomalyDetector.java
│   │   ├── EditingRuleRepair.java, SCREEN.java, Lsgreedy.java, IMR.java, MTCSC.java
│   │   └── util/          # VARUtil, DLinearUtil, PatchTSTUtil, KDTreeUtil, ...
│   └── ...
├── python/                 # Python scripts for DLinear & PatchTST
│   ├── dlinear_model.py
│   ├── patchtst_model.py
│   └── requirements.txt
├── data/                   # Input datasets (time series + master data per domain)
│   ├── engine/, gps/, road/, weather/, traj/
│   └── <domain>/time_series_data_*.csv, master_data_*.csv
├── model/data/             # Output: repaired series, predictions (per run)
├── results/                # Experiment logs: expRMSE.txt, expPrecision.txt, expRecall.txt, expTime.txt, expRegressionLoss.txt
├── fig/                    # Figures (if present)
└── pom.xml
```

## Requirements

- **JDK 17** (or as configured in `pom.xml`)
- **Maven 3.x**
- **Optional (for DLinear/PatchTST)**: Python 3.8+ and the dependencies below.

### Python dependencies (for DLinear & PatchTST)

Install from the project root:

```bash
pip install -r python/requirements.txt
```

| Package | Version |
|---------|---------|
| numpy   | >=1.21.0, &lt;3.0.0 |
| torch   | >=1.12.0 |

For a smaller install (CPU-only PyTorch, no CUDA/MPS):

```bash
pip install "numpy>=1.21.0,<3.0.0"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Build

```bash
mvn clean install -U
```

## Run Experiments

Entry point: `src/main/java/Experiment.java` (run the `main` method).

Configure in `Experiment.java`:

- **Data paths**: `DATA_BASE_PATH` (default `./data/`), `OUTPUT_BASE_PATH` (default `./model/data/`).
- **Which experiment to run** in `main()` (uncomment as needed):

| Method | Description |
|--------|-------------|
| `varyingModel(5)` | VAR model: varying error rate, outputs RMSE/Precision/Recall/Time/RegressionLoss (default). |
| `varyingModelDLinear(5)` | Same with **DLinear** (requires Python). |
| `varyingModelPatchTST(5)` | Same with **PatchTST** (requires Python). |
| `main_td_scale()` | Varying time series length. |
| `main_md_scale()` | Varying master data size. |
| `main_error_rate()` | Varying error rate. |
| `main_error_range()` | Varying error range. |
| `main_error_length()` | Varying error length. |
| `main_parameters(...)` | Varying algorithm parameters. |
| `whole_data_set(4)` | Run on full dataset index. |

Results are appended under `results/` (e.g. `expRMSE.txt`, `expRegressionLoss.txt`). Repaired and prediction series are written under `model/data/<dataset>/` (e.g. `VAR_t.csv`, `DLinear_t.csv`, `DLinear_prediction_t.csv`).

## Data Format

- **Time series**: CSV, one row per timestamp; first column time, remaining columns are variables.
- **Master data**: CSV, same variable columns (no time column). Each row is a “clean” reference tuple.

Example (trajectory):

- `data/traj/time_series_data_25168.csv` — time series.
- `data/traj/master_data_1180.csv` — master data.

Dataset names and paths are set in `Experiment.init(dataset_idx)` for indices 0–4 (engine, gps, road, weather, traj).

## Prediction Models

- **VAR** (default): Pure Java (`VARUtil`), no Python. Uses normalized VAR and k-NN over master data for repair.
- **DLinear**: Python script `python/dlinear_model.py`; train with `fit`, predict with `predict` or interactive process. Requires Python + PyTorch.
- **PatchTST**: Python script `python/patchtst_model.py`; same interface. Requires Python + PyTorch.

To use DLinear/PatchTST:

1. Install [Python dependencies](#python-dependencies-for-dlinear--patchtst) above.
2. From project root, run Java; it will call the scripts under `python/` (paths are configurable in the util classes).

## Deployment

The method has been integrated into **Apache IoTDB** for anomaly detection with repair:

- [IoTDB UDF – MasterDetector](https://iotdb.apache.org/UserGuide/latest/SQL-Manual/UDF-Libraries_apache.html#masterdetect)
- [IoTDB research/master-detector UDF source](https://github.com/apache/iotdb/tree/research/master-detector/library-udf/src/main/java/org/apache/iotdb/library/anomaly)

## License

See repository or project root for license information.
