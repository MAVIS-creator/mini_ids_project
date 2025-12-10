Mini Intrusion Detection System (Mini IDS)

Project objective
- Classify network traffic as `normal` or `attack` using sample features: `duration`, `bytes`, and `protocol`.
- Algorithm: Random Forest Classifier
- Expected outputs: confusion matrix image and a `metrics.txt` with accuracy, precision, recall.

Files created
- `data_utils.py` : CSV loading and preprocessing (selects/creates `bytes`, encodes `protocol`).
- `train.py` : trains RandomForest and writes `model.joblib`.
- `evaluate.py` : evaluates a saved model on test CSV and writes results to `results/`.
- `run.py` : convenience runner to train and evaluate with defaults.
- `requirements.txt` : Python dependencies.

Usage (Windows cmd / PowerShell)

1. Install dependencies (use your Python environment):

```powershell
python -m pip install -r mini_ids_project\requirements.txt
```

2. Run training + evaluation:

```powershell
python mini_ids_project\run.py --train kddtrain.csv --test kddtest.csv
```

Outputs
- `mini_ids_project/model.joblib` : saved model and protocol encoder.
- `mini_ids_project/results/metrics.txt` : accuracy, precision, recall.
- `mini_ids_project/results/confusion_matrix.png` : confusion matrix image.

Notes
- The code looks for `duration`, `protocol_type` or `protocol`, and `src_bytes`/`dst_bytes` (or `bytes`) in the CSVs. If column names differ, rename columns or edit `data_utils.py` accordingly.
- Labels are mapped to binary: any label containing `normal` (case-insensitive) is treated as normal (0); all others as attack (1).
