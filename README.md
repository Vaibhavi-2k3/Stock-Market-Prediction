# Stock Market Prediction

Compare **linear regression** and a **Long Short-Term Memory (LSTM)** neural
network on historical Tata Consultancy Services (TCS) stock prices. Both models
use chronological training/test splits and produce evaluation metrics, charts,
and CSV files of held-out predictions.

## Features

- Bundled TCS dataset covering January 2004 through April 2024.
- Linear regression baseline using calendar dates as the input feature.
- Two-layer LSTM using 60 trading observations of closing-price history.
- Training-only scaler fitting to prevent test-data leakage.
- MSE, RMSE, R-squared, and a previous-close prediction baseline.
- Saved charts and prediction CSVs, with optional headless execution.

These are standalone Python scripts; no web server or API key is required.

## Project structure

```text
Stock-Market-Prediction/
|-- linear_model.py       # Linear regression training and evaluation
|-- lstm_model.py         # LSTM training and evaluation
|-- TCS.NS.csv            # Historical stock-price dataset
|-- requirements.txt     # Pinned Python dependencies
|-- README.md
|-- LICENSE
`-- outputs/             # Generated locally when a model runs
```

## Local setup

### Windows / PowerShell

Use Python 3.11 and Git:

```powershell
git clone https://github.com/Vaibhavi-2k3/Stock-Market-Prediction.git
cd Stock-Market-Prediction
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe linear_model.py
.\.venv\Scripts\python.exe lstm_model.py --epochs 5
```

### Linux / macOS

```bash
git clone https://github.com/Vaibhavi-2k3/Stock-Market-Prediction.git
cd Stock-Market-Prediction
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python linear_model.py
.venv/bin/python lstm_model.py --epochs 5
```

The project was verified on Windows with Python 3.11. TensorFlow runs on the CPU;
the LSTM takes longer than linear regression. Use the virtual environment's
Python executable so the pinned dependencies are selected.

## Run options

Each script saves a plot and a CSV of held-out predictions in `outputs/`, then
opens its plot window. For an unattended run that only saves results:

```powershell
.\.venv\Scripts\python.exe linear_model.py --no-show
.\.venv\Scripts\python.exe lstm_model.py --epochs 5 --no-show
```

Use `--output-dir PATH` to choose another output directory. The input CSV is
resolved relative to the script, so execution also works from other directories.

### Generated files

| File | Contents |
| --- | --- |
| `outputs/linear_prediction.png` | Actual prices and linear regression predictions |
| `outputs/linear_predictions.csv` | Test dates, actual prices, and predicted prices |
| `outputs/lstm_prediction.png` | Actual prices and LSTM predictions over the test period |
| `outputs/lstm_predictions.csv` | Test dates, actual prices, and predicted prices |

The virtual environment and generated outputs are excluded from version control.

## Models and evaluation

- **Linear regression:** learns closing price from the ordinal calendar date;
  trains on the first 80% of valid rows and evaluates on the final 20%.
- **LSTM:** two 50-unit recurrent layers with dropout, using the previous 60
  trading observations to predict the next closing price. Splits sequences
  chronologically 80/20; the scaler is fitted only on training history. Training
  uses a fixed seed and five epochs by default. Training batches are shuffled;
  the observations within each sequence remain chronological.
- Both print MSE, RMSE, R-squared, and a previous-close baseline RMSE. The models'
  test periods differ slightly because LSTM needs 60 observations of history.
- LSTM evaluation is rolling one-step-ahead prediction: later test predictions
  use already observed test-period prices. It is not a multi-day future forecast.

## Example results

Results from a local run with five LSTM epochs and random seed 42:

| Model | Test observations | RMSE (INR; lower is better) | R-squared | Previous-close baseline RMSE (INR) |
| --- | ---: | ---: | ---: | ---: |
| Linear regression | 1,002 | 1,234.97 | -4.593 | 45.41 |
| LSTM | 990 | 308.50 | 0.619 | 45.09 |

The LSTM tracks historical price levels better than the linear model in this
run, but **both models perform worse than the previous-close baseline**.
The test periods differ slightly, so these are not strictly like-for-like
comparisons. Neural-network results may vary with hardware and numerical
execution details even with a fixed seed.

## Dataset observations

The CSV contains 5,017 rows from 2004-01-16 through 2024-04-16. Eight rows have
missing prices and are removed, leaving 5,009 observations with unique,
chronologically ordered dates. There are 91 zero-volume observations among the
valid rows; data quality should be reviewed before further modeling.

The scripts use `Close`, not `Adj Close`, and evaluate historical data rather
than downloading live prices. A high R-squared on price levels alone does not
establish useful predictive performance; compare errors against the printed
previous-close baseline.

## Troubleshooting

If TensorFlow fails with `cannot import name 'runtime_version' from
'google.protobuf'`, install `requirements.txt` using the virtual environment's
Python executable. The requirements include a compatible Protobuf version.

On a machine without a graphical display, use `--no-show` to save the plots.

## License and attribution

Distributed under the [MIT License](LICENSE). The original license credits
Sarthak Chauhan (2025); that attribution is retained. This version adds
reproducible setup instructions, command-line options, saved outputs, expanded
evaluation, and training-only scaling for the LSTM.
