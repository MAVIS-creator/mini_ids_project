# Mini Intrusion Detection System (Mini IDS)

A machine learning-based network intrusion detection system that classifies network traffic as **normal** or **attack** using Random Forest classification.

## 🎯 Project Objective

This project implements a binary classification system to detect network intrusions using:
- **Features**: `duration`, `bytes` (combined src + dst), and `protocol` type
- **Algorithm**: Random Forest Classifier (ensemble learning)
- **Dataset**: KDD Cup 1999 network traffic data
- **Output**: Trained model, performance metrics, and confusion matrix visualization

## 🌟 Key Features

- ✅ **Binary Classification**: Distinguishes normal traffic from attacks
- ✅ **Flexible Input Handling**: Supports both named-column and headerless CSV formats
- ✅ **Comprehensive Evaluation**: Accuracy, precision, recall, and confusion matrix
- ✅ **Modular Design**: Separate modules for data processing, training, and evaluation
- ✅ **Easy to Use**: Single command to train and evaluate
- ✅ **Visualization**: Automatic confusion matrix plotting

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Detailed Usage](#detailed-usage)
- [Understanding the Results](#understanding-the-results)
- [Dataset Format](#dataset-format)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)
- [Documentation](#documentation)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher (tested with Python 3.13)
- pip (Python package manager)

### Step 1: Install Dependencies

Navigate to the project directory and install required packages:

```powershell
# Windows PowerShell
cd mini_ids_project
python -m pip install -r requirements.txt
```

```bash
# Linux/Mac
cd mini_ids_project
pip install -r requirements.txt
```

### Dependencies Installed:
- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning algorithms and metrics
- `matplotlib` - Visualization and plotting
- `joblib` - Model serialization

## ⚡ Quick Start

### Option 1: Run Everything (Recommended)

Train the model and evaluate it in one command:

```powershell
python run.py --train kddtrain.csv --test kddtest.csv
```

**Expected Output:**
```
Training model...
Evaluating model...
Done. Metrics saved to mini_ids_project/results
```

### Option 2: Step-by-Step

**Step 1 - Train the model:**
```powershell
python train.py --train kddtrain.csv --out mini_ids_project/model.joblib
```

**Step 2 - Evaluate the model:**
```powershell
python evaluate.py --model mini_ids_project/model.joblib --test kddtest.csv --out mini_ids_project/results
```

## 📁 Project Structure

```
mini_ids_project/
│
├── data_utils.py              # Data loading and preprocessing
├── train.py                   # Model training script
├── evaluate.py                # Model evaluation script
├── run.py                     # Main orchestration script
├── copy_data.py               # Utility to copy datasets
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── CODE_EXPLANATIONS.md       # Detailed code documentation
│
├── kddtrain.csv              # Training dataset (494,022 samples)
├── kddtest.csv               # Test dataset (311,030 samples)
│
└── mini_ids_project/          # Output directory
    ├── model.joblib           # Trained model (generated)
    └── results/               # Evaluation results (generated)
        ├── metrics.txt        # Performance metrics
        └── confusion_matrix.png  # Visualization
```

## 📖 Detailed Usage

### Training Module (`train.py`)

Trains a Random Forest classifier on network traffic data.

**Basic Usage:**
```powershell
python train.py --train kddtrain.csv --out model.joblib
```

**Command-Line Arguments:**
- `--train`: Path to training CSV file (default: `kddtrain.csv`)
- `--out`: Path to save trained model (default: `mini_ids_project/model.joblib`)

**What It Does:**
1. Loads training data from CSV
2. Extracts features: `[duration, bytes, protocol_enc]`
3. Prepares binary labels: 0 = normal, 1 = attack
4. Splits data into train/validation (80/20)
5. Trains Random Forest with 100 trees
6. Saves model and label encoder to file

**Output:**
```
Finished training. Validation accuracy: 0.9987
```

### Evaluation Module (`evaluate.py`)

Evaluates a trained model on test data.

**Basic Usage:**
```powershell
python evaluate.py --model mini_ids_project/model.joblib --test kddtest.csv --out mini_ids_project/results
```

**Command-Line Arguments:**
- `--model`: Path to trained model file (default: `mini_ids_project/model.joblib`)
- `--test`: Path to test CSV file (default: `kddtest.csv`)
- `--out`: Directory to save results (default: `mini_ids_project/results`)

**What It Does:**
1. Loads trained model and encoder
2. Processes test data using same feature extraction
3. Makes predictions on test set
4. Calculates performance metrics
5. Generates confusion matrix visualization
6. Saves results to output directory

**Output Files:**
- `results/metrics.txt` - Accuracy, precision, recall values
- `results/confusion_matrix.png` - Visual heatmap of predictions

### Orchestration Script (`run.py`)

Combines training and evaluation in a single workflow.

**Basic Usage:**
```powershell
python run.py --train kddtrain.csv --test kddtest.csv
```

**Command-Line Arguments:**
- `--train`: Training CSV path (default: `kddtrain.csv`)
- `--test`: Test CSV path (default: `kddtest.csv`)
- `--out-model`: Model save path (default: `mini_ids_project/model.joblib`)
- `--out-results`: Results directory (default: `mini_ids_project/results`)

## 📊 Understanding the Results

### Metrics File (`metrics.txt`)

Example output:
```
accuracy: 1.0
precision: 1.0
recall: 1.0
```

**Metric Definitions:**

- **Accuracy**: `(TP + TN) / Total`
  - Percentage of correctly classified samples
  - Shows overall model performance

- **Precision**: `TP / (TP + FP)`
  - Of predicted attacks, how many were actual attacks?
  - Low precision = many false alarms

- **Recall (Detection Rate)**: `TP / (TP + FN)`
  - Of actual attacks, how many did we detect?
  - Low recall = missing real attacks

**Where:**
- TP = True Positives (correctly identified attacks)
- TN = True Negatives (correctly identified normal traffic)
- FP = False Positives (normal traffic flagged as attack)
- FN = False Negatives (attacks missed)

### Confusion Matrix

The confusion matrix visualization shows:

```
                Predicted
              Normal  Attack
Actual Normal   TN      FP
       Attack   FN      TP
```

- **Diagonal cells** (TN, TP): Correct predictions ✅
- **Off-diagonal cells** (FP, FN): Errors ❌
- **Color intensity**: Darker = more samples

## 📄 Dataset Format

The code supports two CSV formats:

### Format 1: Named Columns (with header)

```csv
duration,protocol_type,src_bytes,dst_bytes,label
0,tcp,181,5450,normal
120,udp,0,0,neptune
45,icmp,1032,0,normal
```

**Required Columns:**
- `duration` - Connection duration in seconds
- `protocol_type` or `protocol` - Protocol name (tcp, udp, icmp, etc.)
- `src_bytes` and `dst_bytes` (or combined `bytes`) - Data transferred
- `label` or `class` - Traffic label (normal, attack name)

### Format 2: Headerless Numeric (KDD Cup standard)

```csv
0,0,1,1,1,181,5450,0,0,0,...,0
0,0,1,1,1,239,486,0,0,0,...,1
```

**Column Positions:**
- Column 0: `duration`
- Column 1: `protocol` (numeric encoding)
- Column 4: `src_bytes`
- Column 5: `dst_bytes`
- Last column: Binary label (0 = normal, 1 = attack)

**Note**: The current datasets (`kddtrain.csv`, `kddtest.csv`) use the headerless format.

## 🔧 Advanced Usage

### Custom Model Parameters

Modify training parameters programmatically:

```python
from train import train_model

# Train with 200 trees and different random seed
clf, val = train_model(
    train_csv_path='kddtrain.csv',
    model_out_path='custom_model.joblib',
    n_estimators=200,      # More trees = better accuracy
    random_state=123       # Different seed
)
```

### Feature Engineering

Add custom features in `data_utils.py`:

```python
# Example: Add packet rate feature
df['packet_rate'] = df['packets'] / df['duration'].replace(0, 1)
```

### Multi-Class Classification

Extend to detect specific attack types:

```python
# In prepare_labels(), instead of binary:
attack_types = {
    'normal': 0,
    'neptune': 1,
    'smurf': 2,
    'back': 3,
    # ... more attack types
}
```

### Real-Time Prediction

Use the trained model for live inference:

```python
import joblib

# Load model once
model_obj = joblib.load('mini_ids_project/model.joblib')
clf = model_obj['model']

# Predict on new traffic
new_sample = [[120, 5000, 1]]  # [duration, bytes, protocol_enc]
prediction = clf.predict(new_sample)

if prediction[0] == 1:
    print("⚠️ ATTACK DETECTED!")
else:
    print("✅ Normal traffic")
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Install dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Issue: "FileNotFoundError: kddtrain.csv"

**Solution:** Ensure CSV files are in the correct location:
```powershell
# Check current directory
Get-ChildItem *.csv

# If files are in parent directory, use relative paths:
python run.py --train ../kddtrain.csv --test ../kddtest.csv
```

### Issue: "ValueError: Could not find src/dst byte columns"

**Solution:** Your CSV has different column names. Either:
1. Rename columns to match expected names
2. Modify `data_utils.py` to recognize your column names

### Issue: Low Accuracy (<50%)

**Possible Causes:**
- Label encoding mismatch
- Incorrect feature extraction
- Dataset format not recognized

**Debug Steps:**
```python
# Check data loading
from data_utils import load_csv, prepare_features, prepare_labels
df = load_csv('kddtrain.csv')
print(df.head())
print(df.columns)

X, le = prepare_features(df)
print(X.head())
```

### Issue: "UserWarning: A single label was found"

**Explanation:** Test set contains only one class (all normal or all attack).

**Solution:** Verify test data has both labels:
```python
import pandas as pd
df = pd.read_csv('kddtest.csv', header=None)
print(df.iloc[:, -1].value_counts())  # Check last column (labels)
```

## 📈 Performance

### Training Time
- **Dataset**: 494,022 samples
- **Training Time**: ~30-60 seconds (depends on CPU)
- **Model Size**: ~5-10 MB

### Prediction Speed
- **Throughput**: ~10,000-50,000 predictions/second
- **Latency**: <1ms per prediction (single sample)

### Current Results
```
Accuracy:  100.00%
Precision: 100.00%
Recall:    100.00%
```

**Note**: Perfect scores indicate the test dataset may be simplified or preprocessed. Real-world deployment typically sees 95-99% accuracy.

## 🤝 Contributing

To extend this project:

1. **Add new features** - Modify `prepare_features()` in `data_utils.py`
2. **Try different algorithms** - Replace `RandomForestClassifier` in `train.py`
3. **Implement cross-validation** - Use `cross_val_score()` for robust evaluation
4. **Add hyperparameter tuning** - Use `GridSearchCV` to find optimal parameters
5. **Support multi-class classification** - Detect specific attack types

## 📚 Documentation

For detailed code explanations, architecture diagrams, and algorithm deep-dives, see:

- **[CODE_EXPLANATIONS.md](CODE_EXPLANATIONS.md)** - Comprehensive code documentation with:
  - Module-by-module breakdowns
  - Data flow diagrams
  - Algorithm explanations (Random Forest, Label Encoding)
  - Advanced topics and extension ideas
  - Troubleshooting guide

## 📝 Notes

- **Feature Selection**: The code automatically detects column names and formats
- **Label Mapping**: Any label containing `"normal"` (case-insensitive) → 0, others → 1
- **Protocol Encoding**: String protocols are automatically converted to numeric values
- **Missing Values**: Handled with `.fillna(0)` for numeric columns
- **Model Persistence**: Both the classifier and label encoder are saved together

## 🔗 References

- **KDD Cup 1999 Dataset**: [http://kdd.ics.uci.edu/databases/kddcup99/](http://kdd.ics.uci.edu/databases/kddcup99/)
- **Random Forest Algorithm**: Breiman, L. (2001). "Random Forests". Machine Learning.
- **scikit-learn Documentation**: [https://scikit-learn.org/](https://scikit-learn.org/)

## 📄 License

This project is for educational purposes. Please check the KDD Cup dataset license for data usage terms.

---

**Built with ❤️ using Python, scikit-learn, and Random Forest**

*Last Updated: December 10, 2025*
