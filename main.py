"""
Mini Intrusion Detection System (Mini IDS) - Consolidated Version
A machine learning-based network intrusion detection system using Random Forest.

Usage:
    python main.py --train kddtrain.csv --test kddtest.csv [--out-model MODEL_PATH] [--out-results RESULTS_DIR]
    python main.py train --train kddtrain.csv --out MODEL_PATH
    python main.py evaluate --model MODEL_PATH --test kddtest.csv --out RESULTS_DIR
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_csv(path, nrows=None, header='infer'):
    """Load CSV file with flexible header detection."""
    return pd.read_csv(path, header=header, nrows=nrows)


def prepare_features(df):
    """
    Extract and prepare features from dataset.
    
    Supports two formats:
    1. Named columns: duration, protocol_type/protocol, src_bytes/dst_bytes
    2. Headerless KDD numeric: columns 0, 1, 4, 5 for duration, protocol, src/dst bytes
    
    Returns:
        X (DataFrame): Feature matrix with [duration, bytes, protocol_enc]
        le (LabelEncoder or None): Label encoder for protocol (None if already numeric)
    """
    df = df.copy()

    # Format 1: Named columns
    if 'duration' in df.columns:
        # Find byte columns
        src_candidates = ['src_bytes', 'src-bytes', 'src bytes']
        dst_candidates = ['dst_bytes', 'dst-bytes', 'dst bytes']
        src_col = next((c for c in src_candidates if c in df.columns), None)
        dst_col = next((c for c in dst_candidates if c in df.columns), None)

        if src_col and dst_col:
            df['bytes'] = df[src_col].fillna(0) + df[dst_col].fillna(0)
        elif src_col:
            df['bytes'] = df[src_col].fillna(0)
        elif dst_col:
            df['bytes'] = df[dst_col].fillna(0)
        else:
            if 'bytes' in df.columns:
                df['bytes'] = df['bytes'].fillna(0)
            else:
                raise ValueError('Could not find src/dst byte columns')

        proto_col = 'protocol_type' if 'protocol_type' in df.columns else ('protocol' if 'protocol' in df.columns else None)
        if proto_col is None:
            raise ValueError('Expected protocol column `protocol_type` or `protocol`')

        features = df[['duration', 'bytes', proto_col]].copy()
        features = features.rename(columns={proto_col: 'protocol'})

        le = LabelEncoder()
        features['protocol_enc'] = le.fit_transform(features['protocol'].astype(str))
        X = features[['duration', 'bytes', 'protocol_enc']].astype(float)
        return X, le

    # Format 2: Headerless KDD numeric
    if df.shape[1] >= 6:
        df.columns = list(range(df.shape[1]))
        duration = df[0].astype(float).fillna(0)
        bytes_col = df[4].fillna(0).astype(float) + df[5].fillna(0).astype(float)
        protocol_enc = df[1].astype(int).fillna(0)

        X = pd.DataFrame({'duration': duration, 'bytes': bytes_col, 'protocol_enc': protocol_enc})
        return X, None

    raise ValueError('Unrecognized dataset format for feature extraction')


def prepare_labels(df):
    """Extract and prepare binary labels (0=normal, 1=attack)."""
    # Named label column
    for name in ['label', 'class']:
        if name in df.columns:
            labels = df[name].astype(str)
            y = labels.apply(lambda v: 0 if 'normal' in v.lower() else 1).values
            return y

    # Numeric label (last column)
    if df.shape[1] >= 1:
        last_col = df.columns[-1]
        try:
            vals = df[last_col].astype(float).fillna(0)
            unique_vals = set(np.unique(vals))
            if unique_vals.issubset({0.0, 1.0}):
                return vals.astype(int).values
        except Exception:
            pass

        labels = df[last_col].astype(str)
        y = labels.apply(lambda v: 0 if 'normal' in v.lower() else 1).values
        return y

    raise ValueError('Could not determine label column')


# ============================================================================
# TRAINING
# ============================================================================

def train_model(train_csv_path, model_out_path='model.joblib', n_estimators=100, random_state=42):
    """
    Train Random Forest classifier on network traffic data.
    
    Args:
        train_csv_path: Path to training CSV
        model_out_path: Where to save trained model
        n_estimators: Number of trees (default: 100)
        random_state: Random seed (default: 42)
        
    Returns:
        clf: Trained classifier
        val_set: Validation (X_val, y_val) tuple
    """
    print(f"Loading training data from {train_csv_path}...")
    df = load_csv(train_csv_path)
    
    print("Preparing features and labels...")
    X, le = prepare_features(df)
    y = prepare_labels(df)

    # Train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    print(f"Training Random Forest ({n_estimators} trees)...")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Save model and encoder
    os.makedirs(os.path.dirname(model_out_path) or '.', exist_ok=True)
    joblib.dump({'model': clf, 'protocol_le': le}, model_out_path)
    print(f"Model saved to {model_out_path}")

    # Validation accuracy
    val_acc = clf.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}\n")
    
    return clf, (X_val, y_val)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_path, test_csv_path, out_dir='results'):
    """
    Evaluate trained model on test data.
    
    Args:
        model_path: Path to saved model
        test_csv_path: Path to test CSV
        out_dir: Directory to save results
        
    Returns:
        Dictionary with accuracy, precision, recall, confusion_matrix
    """
    print(f"Loading model from {model_path}...")
    obj = joblib.load(model_path)
    clf = obj['model']
    le = obj.get('protocol_le', None)

    print(f"Loading test data from {test_csv_path}...")
    df = load_csv(test_csv_path)
    X, _ = prepare_features(df)
    y = prepare_labels(df)

    print("Making predictions...")
    preds = clf.predict(X)

    # Calculate metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    
    print("\nEvaluation Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}\n")

    # Save metrics to file
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f'accuracy: {acc}\n')
        f.write(f'precision: {prec}\n')
        f.write(f'recall: {rec}\n')

    # Generate confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Network Intrusion Detection')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['normal', 'attack'])
    plt.yticks(tick_marks, ['normal', 'attack'])

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black', fontsize=14, fontweight='bold')

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=100)
    plt.close()

    print(f"Metrics saved to {os.path.join(out_dir, 'metrics.txt')}")
    print(f"Confusion matrix saved to {cm_path}\n")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'confusion_matrix': cm}


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main_combined(train_path, test_path, model_out, results_out):
    """Run training and evaluation pipeline."""
    print("="*70)
    print("MINI INTRUSION DETECTION SYSTEM - Training & Evaluation Pipeline")
    print("="*70 + "\n")
    
    train_model(train_path, model_out)
    evaluate_model(model_out, test_path, results_out)
    
    print("="*70)
    print(f"✓ Pipeline complete!")
    print(f"  Model:   {model_out}")
    print(f"  Results: {results_out}")
    print("="*70)


def main_train_only(train_path, model_out, n_estimators=100):
    """Run training only."""
    print("="*70)
    print("MINI INTRUSION DETECTION SYSTEM - Training Only")
    print("="*70 + "\n")
    
    clf, (X_val, y_val) = train_model(train_path, model_out, n_estimators)
    
    print("="*70)
    print(f"✓ Training complete!")
    print(f"  Model: {model_out}")
    print("="*70)


def main_eval_only(model_path, test_path, results_out):
    """Run evaluation only."""
    print("="*70)
    print("MINI INTRUSION DETECTION SYSTEM - Evaluation Only")
    print("="*70 + "\n")
    
    evaluate_model(model_path, test_path, results_out)
    
    print("="*70)
    print(f"✓ Evaluation complete!")
    print(f"  Results: {results_out}")
    print("="*70)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Mini Intrusion Detection System - Train and evaluate network traffic classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train and evaluate:
    python main.py --train kddtrain.csv --test kddtest.csv
  
  Train only:
    python main.py train --train kddtrain.csv --out model.joblib
  
  Evaluate only:
    python main.py evaluate --model model.joblib --test kddtest.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Default command (train + evaluate)
    parser.add_argument('--train', default='kddtrain.csv', help='Training CSV path')
    parser.add_argument('--test', default='kddtest.csv', help='Test CSV path')
    parser.add_argument('--out-model', default='model.joblib', help='Model output path')
    parser.add_argument('--out-results', default='results', help='Results directory')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train model only')
    train_parser.add_argument('--train', required=True, help='Training CSV path')
    train_parser.add_argument('--out', default='model.joblib', help='Model output path')
    train_parser.add_argument('--trees', type=int, default=100, help='Number of trees (default: 100)')

    # Evaluate subcommand
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model only')
    eval_parser.add_argument('--model', required=True, help='Model path')
    eval_parser.add_argument('--test', required=True, help='Test CSV path')
    eval_parser.add_argument('--out', default='results', help='Results directory')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        if args.command == 'train':
            main_train_only(args.train, args.out, args.trees)
        elif args.command == 'evaluate':
            main_eval_only(args.model, args.test, args.out)
        else:
            # Default: train and evaluate
            main_combined(args.train, args.test, args.out_model, args.out_results)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
