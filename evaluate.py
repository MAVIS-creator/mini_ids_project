import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from data_utils import load_csv, prepare_features, prepare_labels


def evaluate_model(model_path, test_csv_path, out_dir='results'):
    obj = joblib.load(model_path)
    clf = obj['model']
    le = obj.get('protocol_le', None)

    df = load_csv(test_csv_path)
    X, _ = prepare_features(df)
    y = prepare_labels(df)

    preds = clf.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)

    os.makedirs(out_dir, exist_ok=True)
    # save metrics
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f'accuracy: {acc}\n')
        f.write(f'precision: {prec}\n')
        f.write(f'recall: {rec}\n')

    # save confusion matrix plot
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['normal', 'attack'])
    plt.yticks(tick_marks, ['normal', 'attack'])

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'confusion_matrix': cm}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Mini IDS model')
    parser.add_argument('--model', default='mini_ids_project/model.joblib', help='Path to saved model')
    parser.add_argument('--test', default='kddtest.csv', help='Path to test CSV')
    parser.add_argument('--out', default='mini_ids_project/results', help='Output directory')
    args = parser.parse_args()

    res = evaluate_model(args.model, args.test, args.out)
    print('Evaluation results:')
    print('Accuracy', res['accuracy'])
    print('Precision', res['precision'])
    print('Recall', res['recall'])
