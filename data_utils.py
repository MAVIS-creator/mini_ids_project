import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings


def load_csv(path, nrows=None, header='infer'):
    # Try to read with inferred header first; caller can pass header=None
    return pd.read_csv(path, header=header, nrows=nrows)


def prepare_features(df):
    df = df.copy()

    # If dataset has named columns (e.g., contains 'duration'), use names.
    if 'duration' in df.columns:
        # Bytes: prefer `src_bytes` and `dst_bytes` then fallback to common variants
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

    # If no named columns, assume classic KDD numeric format (no header):
    # indices: 0=duration, 1=protocol (encoded numeric), 4=src_bytes, 5=dst_bytes, last=label
    if df.shape[1] >= 6:
        # Ensure columns are numbered 0..n-1
        df.columns = list(range(df.shape[1]))
        duration = df[0].astype(float).fillna(0)

        # bytes
        if 4 in df.columns and 5 in df.columns:
            bytes_col = df[4].fillna(0).astype(float) + df[5].fillna(0).astype(float)
        else:
            warnings.warn('Expected src_bytes/dst_bytes at columns 4 and 5; falling back to column 4 only if present')
            if 4 in df.columns:
                bytes_col = df[4].fillna(0).astype(float)
            else:
                bytes_col = np.zeros(len(df))

        # protocol (numeric encoding)
        if 1 in df.columns:
            protocol_enc = df[1].astype(int).fillna(0)
        else:
            protocol_enc = np.zeros(len(df), dtype=int)

        X = pd.DataFrame({'duration': duration, 'bytes': bytes_col, 'protocol_enc': protocol_enc})
        return X, None

    raise ValueError('Unrecognized dataset format for feature extraction')


def prepare_labels(df):
    # If named label column exists
    for name in ['label', 'class']:
        if name in df.columns:
            labels = df[name].astype(str)
            y = labels.apply(lambda v: 0 if 'normal' in v.lower() else 1).values
            return y

    # If numeric label is the last column (common in headerless KDD numeric files)
    if df.shape[1] >= 1:
        last_col = df.columns[-1]
        # If numeric 0/1, use directly
        try:
            vals = df[last_col].astype(float).fillna(0)
            # if values are 0/1, take as is
            unique_vals = set(np.unique(vals))
            if unique_vals.issubset({0.0, 1.0}):
                return vals.astype(int).values
        except Exception:
            pass

        # Otherwise convert textual labels containing 'normal' to 0
        labels = df[last_col].astype(str)
        y = labels.apply(lambda v: 0 if 'normal' in v.lower() else 1).values
        return y

    raise ValueError('Could not determine label column')
