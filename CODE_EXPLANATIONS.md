# Mini IDS Project - Code Explanations

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Module Descriptions](#module-descriptions)
4. [Data Flow](#data-flow)
5. [Key Algorithms](#key-algorithms)
6. [Usage Examples](#usage-examples)

---

## Project Overview

The Mini Intrusion Detection System (Mini IDS) is a machine learning-based network traffic classifier that identifies whether network traffic is **normal** or an **attack**. It uses a **Random Forest Classifier** from scikit-learn trained on the KDD Cup dataset.

### Key Features:
- Binary classification: `normal` (0) vs `attack` (1)
- Features: `duration`, `bytes` (src + dst), `protocol` (encoded)
- Algorithm: Random Forest (ensemble of decision trees)
- Outputs: Trained model, confusion matrix visualization, performance metrics

### Technologies Used:
- **Python 3.13+**
- **pandas**: Data loading and manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning (Random Forest, metrics)
- **matplotlib**: Visualization (confusion matrix plot)
- **joblib**: Model serialization

---

## Architecture

```
┌─────────────────┐
│  kddtrain.csv   │──────┐
│  kddtest.csv    │      │
└─────────────────┘      │
                         ▼
                  ┌─────────────┐
                  │ data_utils  │ (Feature extraction & label processing)
                  └─────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼                             ▼
    ┌──────────┐                 ┌─────────────┐
    │  train   │────────────────▶│ model.joblib│
    └──────────┘                 └─────────────┘
                                        │
                                        ▼
                                 ┌─────────────┐
                                 │  evaluate   │
                                 └─────────────┘
                                        │
                       ┌────────────────┼────────────────┐
                       ▼                                 ▼
              ┌─────────────────┐              ┌──────────────────┐
              │  metrics.txt    │              │ confusion_matrix │
              │ (accuracy, etc.)│              │      .png        │
              └─────────────────┘              └──────────────────┘
                                 
    ┌──────────┐
    │  run.py  │ (Orchestrator: calls train → evaluate)
    └──────────┘
```

---

## Module Descriptions

### 1. `data_utils.py` - Data Preprocessing Module

**Purpose**: Load CSV files and prepare features and labels for machine learning.

#### Functions:

##### `load_csv(path, nrows=None, header='infer')`
- **Input**: File path to CSV, optional row limit, header mode
- **Output**: pandas DataFrame
- **Logic**: Wrapper around `pd.read_csv()` with configurable header detection

##### `prepare_features(df)`
- **Input**: pandas DataFrame with network traffic data
- **Output**: Tuple of (X, label_encoder)
  - `X`: Feature matrix with columns `[duration, bytes, protocol_enc]`
  - `label_encoder`: Fitted LabelEncoder for protocol (or None for numeric datasets)

**Feature Extraction Logic**:

1. **Named Column Format** (e.g., columns like `duration`, `protocol_type`, `src_bytes`):
   - Searches for `duration` column
   - Creates `bytes` column by summing `src_bytes` + `dst_bytes` (or fallback to individual columns)
   - Identifies protocol column (`protocol_type` or `protocol`)
   - Encodes protocol strings to integers using `LabelEncoder`
   - Returns 3 features: `[duration, bytes, protocol_enc]`

2. **Headerless Numeric Format** (KDD Cup standard):
   - Assumes positional columns:
     - Column 0: `duration`
     - Column 1: `protocol` (already numeric)
     - Column 4: `src_bytes`
     - Column 5: `dst_bytes`
   - Creates `bytes` = column 4 + column 5
   - Returns `[duration, bytes, protocol_enc]`
   - No LabelEncoder needed (protocol already numeric)

**Why These Features?**:
- **Duration**: Connection time reveals patterns (attacks often have unusual durations)
- **Bytes**: Data volume transferred (attacks may exfiltrate data or send minimal probes)
- **Protocol**: Different protocols have different attack surfaces (TCP, UDP, ICMP)

##### `prepare_labels(df)`
- **Input**: pandas DataFrame
- **Output**: numpy array of binary labels (0 = normal, 1 = attack)

**Label Extraction Logic**:

1. **Named Label Column**: Looks for columns named `label` or `class`
   - Converts to string and checks if `'normal'` appears in lowercase text
   - `'normal'` → 0, everything else → 1

2. **Numeric Last Column** (headerless format):
   - Takes the last column of the DataFrame
   - If already binary (0/1), uses directly
   - Otherwise, applies text-based `'normal'` detection

**Example**:
```python
# Input labels: ['normal', 'neptune', 'normal', 'smurf']
# Output: [0, 1, 0, 1]
```

---

### 2. `train.py` - Model Training Module

**Purpose**: Train a Random Forest classifier and save the trained model.

#### Main Function: `train_model(train_csv_path, model_out_path, n_estimators=100, random_state=42)`

**Parameters**:
- `train_csv_path`: Path to training dataset CSV
- `model_out_path`: Where to save the trained model (default: `model.joblib`)
- `n_estimators`: Number of trees in the Random Forest (default: 100)
- `random_state`: Seed for reproducibility (default: 42)

**Training Pipeline**:

1. **Load Data**:
   ```python
   df = load_csv(train_csv_path)
   ```

2. **Extract Features & Labels**:
   ```python
   X, le = prepare_features(df)  # X: [duration, bytes, protocol_enc]
   y = prepare_labels(df)         # y: [0, 1, 0, 1, ...]
   ```

3. **Train/Validation Split** (80/20):
   ```python
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - Purpose: Internal validation to check model performance during training

4. **Train Random Forest**:
   ```python
   clf = RandomForestClassifier(n_estimators=100, random_state=42)
   clf.fit(X_train, y_train)
   ```

5. **Save Model & Encoder**:
   ```python
   joblib.dump({'model': clf, 'protocol_le': le}, model_out_path)
   ```
   - Saves both the classifier AND the label encoder together
   - This ensures consistent protocol encoding during evaluation

**Why Random Forest?**:
- **Ensemble method**: Combines multiple decision trees → reduces overfitting
- **Robust**: Handles non-linear patterns and feature interactions well
- **Fast**: Efficient training and prediction
- **Interpretable**: Can analyze feature importance

**Standalone Usage**:
```bash
python train.py --train kddtrain.csv --out model.joblib
```

**Output**:
```
Finished training. Validation accuracy: 0.9987
```

---

### 3. `evaluate.py` - Model Evaluation Module

**Purpose**: Evaluate a trained model on test data and generate metrics + visualization.

#### Main Function: `evaluate_model(model_path, test_csv_path, out_dir='results')`

**Parameters**:
- `model_path`: Path to saved model (`model.joblib`)
- `test_csv_path`: Path to test dataset
- `out_dir`: Directory to save results (default: `results/`)

**Evaluation Pipeline**:

1. **Load Trained Model**:
   ```python
   obj = joblib.load(model_path)
   clf = obj['model']
   le = obj.get('protocol_le', None)
   ```

2. **Prepare Test Data**:
   ```python
   df = load_csv(test_csv_path)
   X, _ = prepare_features(df)  # Use SAME feature extraction
   y = prepare_labels(df)
   ```

3. **Predict**:
   ```python
   preds = clf.predict(X)  # [0, 1, 1, 0, ...]
   ```

4. **Calculate Metrics**:
   ```python
   accuracy = accuracy_score(y, preds)      # (TP + TN) / Total
   precision = precision_score(y, preds)    # TP / (TP + FP)
   recall = recall_score(y, preds)          # TP / (TP + FN)
   cm = confusion_matrix(y, preds)          # [[TN, FP], [FN, TP]]
   ```

**Metrics Explained**:
- **Accuracy**: Overall correctness (what % of predictions were correct?)
- **Precision**: Of predicted attacks, how many were actual attacks? (false alarm rate)
- **Recall**: Of actual attacks, how many did we detect? (detection rate)
- **Confusion Matrix**:
  ```
  [[TN  FP]    True Negative  | False Positive
   [FN  TP]]   False Negative | True Positive
  ```

5. **Save Text Metrics**:
   ```python
   # results/metrics.txt
   accuracy: 1.0
   precision: 1.0
   recall: 1.0
   ```

6. **Generate Confusion Matrix Plot**:
   - Heatmap visualization with color intensity
   - Labels: `normal` vs `attack`
   - Cell values show counts
   - Saved as `results/confusion_matrix.png`

**Standalone Usage**:
```bash
python evaluate.py --model model.joblib --test kddtest.csv --out results/
```

---

### 4. `run.py` - Orchestration Script

**Purpose**: Convenience wrapper to run training and evaluation in sequence.

#### Main Function: `main(train_path, test_path, model_out, results_out)`

**Workflow**:

```python
def main(train_path, test_path, model_out, results_out):
    print('Training model...')
    clf, _ = train_model(train_path, model_out)      # Step 1: Train
    
    print('Evaluating model...')
    res = evaluate_model(model_out, test_path, results_out)  # Step 2: Evaluate
    
    print('Done. Metrics saved to', results_out)
```

**Command-Line Arguments**:
- `--train`: Training CSV (default: `kddtrain.csv`)
- `--test`: Test CSV (default: `kddtest.csv`)
- `--out-model`: Model save path (default: `mini_ids_project/model.joblib`)
- `--out-results`: Results directory (default: `mini_ids_project/results`)

**Usage**:
```bash
python run.py --train kddtrain.csv --test kddtest.csv
```

**Output Structure**:
```
mini_ids_project/
├── model.joblib           ← Trained Random Forest + encoder
└── results/
    ├── metrics.txt        ← Accuracy, precision, recall
    └── confusion_matrix.png ← Visualization
```

---

### 5. `copy_data.py` - Data Utility Script

**Purpose**: Copy dataset files from repository root into the project folder.

**Use Case**: If datasets are stored outside the project directory, this script copies them in.

**Logic**:
```python
SRC_TRAIN = '../kddtrain.csv'  # Repository root
SRC_TEST = '../kddtest.csv'
DST_DIR = 'mini_ids_project/'  # Project folder

shutil.copy2(SRC_TRAIN, DST_DIR)
shutil.copy2(SRC_TEST, DST_DIR)
```

**Usage**:
```bash
python copy_data.py
```

---

## Data Flow

### Training Phase:
```
kddtrain.csv (494,022 rows)
    ↓
load_csv()
    ↓
prepare_features() → [duration, bytes, protocol_enc]
prepare_labels()   → [0, 1, 0, 1, ...]
    ↓
train_test_split (80/20)
    ↓
RandomForestClassifier.fit(X_train, y_train)
    ↓
Validate on X_val, y_val
    ↓
Save to model.joblib (classifier + encoder)
```

### Evaluation Phase:
```
kddtest.csv (311,030 rows)
    ↓
load_csv()
    ↓
prepare_features() → [duration, bytes, protocol_enc]
prepare_labels()   → [0, 1, 0, 1, ...]
    ↓
Load model.joblib
    ↓
clf.predict(X_test) → [0, 1, 1, 0, ...]
    ↓
Calculate: accuracy, precision, recall, confusion_matrix
    ↓
Save:
  - results/metrics.txt
  - results/confusion_matrix.png
```

---

## Key Algorithms

### Random Forest Classifier

**Concept**: Ensemble of decision trees that vote on the final prediction.

**How It Works**:

1. **Bootstrap Sampling**:
   - Create N subsets of training data (with replacement)
   - Each tree sees a slightly different dataset

2. **Feature Randomness**:
   - Each split in a tree considers only a random subset of features
   - Reduces correlation between trees

3. **Tree Building**:
   - Each tree grows until leaves are pure (or max depth reached)
   - Split criterion: Gini impurity or entropy

4. **Prediction**:
   - Each tree votes (0 or 1)
   - Final prediction = majority vote

**Example**:
```
Tree 1: predicts 1 (attack)
Tree 2: predicts 1 (attack)
Tree 3: predicts 0 (normal)
...
Tree 100: predicts 1 (attack)

Majority vote: 1 (attack)
```

**Advantages for IDS**:
- Handles non-linear patterns (attacks have complex signatures)
- Resistant to overfitting (averaging many trees)
- Fast inference (parallel tree evaluation)
- Can handle missing values and mixed data types

### Label Encoding

**Purpose**: Convert categorical protocol names to numeric values.

**Example**:
```python
Protocols: ['tcp', 'udp', 'icmp', 'tcp', 'udp']
           ↓ LabelEncoder.fit_transform()
Encoded:   [2, 3, 1, 2, 3]
```

**Why?**: Machine learning algorithms require numeric inputs. LabelEncoder maps strings → integers consistently.

---

## Usage Examples

### Example 1: Basic Run (Recommended)
```bash
# Navigate to project directory
cd mini_ids_project

# Run training + evaluation
python run.py --train kddtrain.csv --test kddtest.csv
```

**Output**:
```
Training model...
Evaluating model...
Done. Metrics saved to mini_ids_project/results
```

### Example 2: Train Only
```bash
python train.py --train kddtrain.csv --out my_model.joblib
```

**Output**:
```
Finished training. Validation accuracy: 0.9987
```

### Example 3: Evaluate Existing Model
```bash
python evaluate.py --model my_model.joblib --test kddtest.csv --out my_results/
```

### Example 4: Custom Parameters
```python
from train import train_model

# Train with 200 trees instead of 100
clf, val = train_model(
    'kddtrain.csv', 
    'model_200trees.joblib',
    n_estimators=200,
    random_state=123
)
```

### Example 5: Programmatic Use
```python
from data_utils import load_csv, prepare_features, prepare_labels
from train import train_model
from evaluate import evaluate_model

# 1. Train
clf, _ = train_model('kddtrain.csv', 'model.joblib')

# 2. Evaluate
results = evaluate_model('model.joblib', 'kddtest.csv', 'results/')

# 3. Access metrics
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
```

---

## Performance Analysis

### Current Results (from last run):
```
Accuracy:  1.0000 (100.00%)
Precision: 1.0000 (100.00%)
Recall:    1.0000 (100.00%)
```

**Interpretation**:
- **Perfect classification** on test set (likely because dataset is simplified/preprocessed)
- **No false positives** (precision = 1.0)
- **No false negatives** (recall = 1.0)
- **Real-world deployment** would likely see lower scores due to:
  - New attack types not in training data
  - Network noise and anomalies
  - Adversarial attacks designed to evade detection

### Warning from Execution:
```
UserWarning: A single label was found in 'y_true' and 'y_pred'.
```

**Meaning**: The test set might contain only one class (all attacks or all normal). This could indicate:
- Dataset preprocessing removed one class
- Need to verify test data has both labels
- Consider using stratified sampling

---

## File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `data_utils.py` | Feature extraction | `load_csv()`, `prepare_features()`, `prepare_labels()` |
| `train.py` | Model training | `train_model()` |
| `evaluate.py` | Model evaluation | `evaluate_model()` |
| `run.py` | Orchestration | `main()` |
| `copy_data.py` | Data copying | `main()` |
| `requirements.txt` | Dependencies | pandas, numpy, scikit-learn, matplotlib, joblib |
| `README.md` | Project documentation | Usage instructions |

---

## Dependencies Explained

### Core Libraries:

1. **pandas** (v2.x recommended):
   - Data loading: `pd.read_csv()`
   - DataFrame operations: column selection, renaming, fillna
   - Efficient CSV parsing

2. **numpy** (v1.x):
   - Numerical arrays
   - Fast array operations
   - Used by scikit-learn internally

3. **scikit-learn** (v1.x):
   - `RandomForestClassifier`: Main ML algorithm
   - `LabelEncoder`: Protocol encoding
   - `train_test_split`: Data splitting
   - `accuracy_score`, `precision_score`, `recall_score`: Metrics
   - `confusion_matrix`: Classification matrix

4. **matplotlib** (v3.x):
   - Plotting: `plt.imshow()`, `plt.savefig()`
   - Confusion matrix visualization
   - Heatmap generation

5. **joblib** (v1.x):
   - Efficient serialization of scikit-learn models
   - Faster than pickle for large numpy arrays
   - Used for saving/loading `model.joblib`

---

## Advanced Topics

### Feature Engineering Ideas:

Current features are minimal. Consider adding:
- **Statistical features**: mean, std, min, max of packet sizes
- **Time-based features**: packets per second, connection rate
- **Protocol flags**: SYN, ACK, FIN flags (TCP)
- **Destination analysis**: unique destination IPs, port scanning patterns

### Model Improvements:

1. **Hyperparameter Tuning**:
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [10, 20, None],
       'min_samples_split': [2, 5, 10]
   }
   
   grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
   grid_search.fit(X_train, y_train)
   best_clf = grid_search.best_estimator_
   ```

2. **Alternative Algorithms**:
   - **Gradient Boosting** (XGBoost, LightGBM): Often outperforms Random Forest
   - **Neural Networks**: Deep learning for complex patterns
   - **Isolation Forest**: Anomaly detection approach

3. **Feature Importance Analysis**:
   ```python
   importances = clf.feature_importances_
   print(f"Duration importance: {importances[0]:.4f}")
   print(f"Bytes importance: {importances[1]:.4f}")
   print(f"Protocol importance: {importances[2]:.4f}")
   ```

### Production Deployment:

For real-world IDS deployment:

1. **Real-time Prediction**:
   ```python
   # Load model once
   model_obj = joblib.load('model.joblib')
   clf = model_obj['model']
   
   # Predict on live traffic
   new_traffic = [[120, 5000, 1]]  # [duration, bytes, protocol]
   prediction = clf.predict(new_traffic)
   if prediction[0] == 1:
       alert("Attack detected!")
   ```

2. **Performance Optimization**:
   - Use `clf.predict_proba()` for confidence scores
   - Batch predictions for efficiency
   - Cache model in memory (don't reload for each prediction)

3. **Model Monitoring**:
   - Track accuracy over time (concept drift)
   - Retrain periodically with new attack samples
   - Monitor false positive/negative rates

---

## Troubleshooting

### Issue: "ValueError: Could not find src/dst byte columns"
**Solution**: Dataset missing expected columns. Check CSV header or adjust `data_utils.py` column mappings.

### Issue: "UserWarning: A single label was found"
**Solution**: Test set contains only one class. Verify both normal and attack samples exist in `kddtest.csv`.

### Issue: Low accuracy (<50%)
**Possible Causes**:
- Label encoding mismatch (encoder not loaded properly)
- Feature scaling needed (try StandardScaler)
- Dataset format changed (verify column order)

### Issue: Model file not found
**Solution**: Ensure training completed successfully before evaluation. Check `model_out_path` parameter.

---

## Contributing

To extend this project:

1. **Add new features**: Modify `prepare_features()` in `data_utils.py`
2. **Try new algorithms**: Replace `RandomForestClassifier` in `train.py`
3. **Add multi-class classification**: Support specific attack types (not just binary)
4. **Implement cross-validation**: Use `cross_val_score()` for robust evaluation
5. **Add logging**: Use Python `logging` module for debugging

---

## References

- **KDD Cup 1999 Dataset**: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- **Random Forest Paper**: Breiman, L. (2001). "Random Forests". Machine Learning.
- **scikit-learn Documentation**: https://scikit-learn.org/stable/
- **Intrusion Detection**: Tavallaee, M., et al. (2009). "A detailed analysis of the KDD CUP 99 data set"

---

## Summary

This Mini IDS project demonstrates:
✅ **End-to-end ML pipeline**: data loading → training → evaluation  
✅ **Modular design**: Separate modules for each task  
✅ **Flexible input handling**: Supports multiple CSV formats  
✅ **Comprehensive evaluation**: Metrics + visualization  
✅ **Production-ready code**: Serialized models, error handling  

**Next Steps**:
- Experiment with more features
- Try ensemble methods (stacking, boosting)
- Deploy as a live network monitor
- Extend to multi-class attack classification

---

*Last Updated: December 10, 2025*
*Project: Mini Intrusion Detection System v1.0*
