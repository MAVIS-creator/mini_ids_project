# Mini Intrusion Detection System (Mini IDS)

A lightweight, consolidated machine learning-based network intrusion detection system that classifies network traffic as **normal** or **attack** using Random Forest classification.

## 🎯 Project Objective

Binary classification system to detect network intrusions using:
- **Features**: duration, bytes (combined src + dst), and protocol type
- **Algorithm**: Random Forest Classifier
- **Dataset**: KDD Cup 1999 network traffic data
- **Output**: Trained model, metrics, and confusion matrix visualization

## 🌟 Key Features

- ✅ **Single Consolidated Script** - All functionality in one `main.py`
- ✅ **Binary Classification** - Normal vs Attack detection
- ✅ **Flexible CSV Support** - Named columns or headerless numeric formats
- ✅ **Complete Metrics** - Accuracy, precision, recall, confusion matrix
- ✅ **Easy to Use** - Simple CLI with train/evaluate modes
- ✅ **Auto Visualization** - Confusion matrix plotting included

## 📋 Quick Navigation

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results & Metrics](#results--metrics)
- [Supported Formats](#supported-formats)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

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

**Dependencies**:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning
- `matplotlib` - Visualization
- `joblib` - Model serialization

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

## ⚡ Usage

### Run Train + Evaluate (Recommended)

```powershell
python main.py --train kddtrain.csv --test kddtest.csv
```

### Train Only

```powershell
python main.py train --train kddtrain.csv --out model.joblib --trees 100
```

### Evaluate Only

```powershell
python main.py evaluate --model model.joblib --test kddtest.csv --out results/
```

## 📁 Project Structure

```
mini_ids_project/
│
├── main.py                    # All-in-one consolidated script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── kddtrain.csv              # Training data (494K samples)
├── kddtest.csv               # Test data (311K samples)
│
└── (Generated on first run)
    ├── model.joblib           # Trained model
    └── results/
        ├── metrics.txt        # Performance metrics
        └── confusion_matrix.png  # Visualization
```

**Total Files: 5 essential files** (down from 11)

## 📖 Detailed Usage

### Training Module

Trains a Random Forest classifier.

```powershell
python main.py train --train kddtrain.csv --out model.joblib --trees 200
```

**Arguments:**
- `--train`: Training CSV file
- `--out`: Model output path (default: `model.joblib`)
- `--trees`: Number of trees (default: 100)

**What happens:**
1. Loads and prepares training data
2. Extracts features: [duration, bytes, protocol]
3. Trains Random Forest (80/20 train/val split)
4. Saves model with encoder

### Evaluation Module

Evaluates trained model on test data.

```powershell
python main.py evaluate --model model.joblib --test kddtest.csv --out results/
```

**Arguments:**
- `--model`: Path to trained model
- `--test`: Test CSV file
- `--out`: Results directory

**Outputs:**
- `results/metrics.txt` - Accuracy, precision, recall
- `results/confusion_matrix.png` - Visualization

## 📊 Results & Metrics

### Output Files

**metrics.txt:**
```
accuracy: 1.0
precision: 1.0
recall: 1.0
```

**Metric Definitions:**
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted attacks, how many were real?
- **Recall**: Of real attacks, how many were detected?

### Confusion Matrix Visualization

Automatically generated `confusion_matrix.png` showing:
- True Positives (TP) - Correct attack detection
- True Negatives (TN) - Correct normal classification
- False Positives (FP) - Normal flagged as attack
- False Negatives (FN) - Missed attacks

## 📄 Supported Formats

### Format 1: Named Columns

```csv
duration,protocol_type,src_bytes,dst_bytes,label
0,tcp,181,5450,normal
120,udp,0,0,neptune
```

### Format 2: Headerless Numeric (KDD Standard)

```csv
0,0,1,1,1,181,5450,0,0,0,...,0
0,0,1,1,1,239,486,0,0,0,...,1
```

The script auto-detects the format.

## � Examples

### Example 1: Complete Pipeline
```powershell
python main.py --train kddtrain.csv --test kddtest.csv
```

### Example 2: Train with More Trees
```powershell
python main.py train --train kddtrain.csv --out my_model.joblib --trees 200
```

### Example 3: Evaluate Existing Model
```powershell
python main.py evaluate --model my_model.joblib --test kddtest.csv --out results/
```

### Example 4: Use in Python Code
```python
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model
model_obj = joblib.load('model.joblib')
clf = model_obj['model']

# Predict on new data
new_sample = [[120, 5000, 1]]  # [duration, bytes, protocol]
prediction = clf.predict(new_sample)

if prediction[0] == 1:
    print("⚠️ Attack detected!")
else:
    print("✅ Normal traffic")
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: kddtrain.csv` | Check CSV is in working directory |
| `ValueError: Could not find src/dst bytes` | Verify column names match expected format |
| Low accuracy | Check dataset format and label column |
| `UserWarning: single label found` | Test set may contain only one class |

## 📈 Performance

- **Training**: ~30-60 seconds (494K samples)
- **Prediction**: ~10K-50K samples/second
- **Model Size**: ~5-10 MB

## 📝 Notes

- Features: [duration, bytes, protocol]
- Labels: 0 = normal, 1 = attack
- Supports named columns or headerless numeric CSVs
- Missing values handled with fillna(0)

## 🔗 References

- [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/)
- [scikit-learn docs](https://scikit-learn.org/)
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)

---

**Built with Python, scikit-learn, and Random Forest**
*Consolidated Version - December 2025*
