import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_utils import load_csv, prepare_features, prepare_labels


def train_model(train_csv_path, model_out_path='model.joblib', n_estimators=100, random_state=42):
    df = load_csv(train_csv_path)
    X, le = prepare_features(df)
    y = prepare_labels(df)

    # small train/val split to report internal validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    # Save model and label encoder together
    os.makedirs(os.path.dirname(model_out_path) or '.', exist_ok=True)
    joblib.dump({'model': clf, 'protocol_le': le}, model_out_path)

    # Return classifier and validation set for quick checks
    return clf, (X_val, y_val)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Random Forest for Mini IDS')
    parser.add_argument('--train', default='kddtrain.csv', help='Path to training CSV')
    parser.add_argument('--out', default='mini_ids_project/model.joblib', help='Model output path')
    args = parser.parse_args()

    clf, val = train_model(args.train, args.out)
    X_val, y_val = val
    print('Finished training. Validation accuracy: {:.4f}'.format(clf.score(X_val, y_val)))
