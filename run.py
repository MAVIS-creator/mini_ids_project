"""Simple runner for the Mini IDS project.

Usage examples (PowerShell / cmd):

python mini_ids_project\run.py --train kddtrain.csv --test kddtest.csv

This will train a RandomForest and evaluate it, saving model to `mini_ids_project/model.joblib`
and results to `mini_ids_project/results/`.
"""
import argparse
import os
from train import train_model
from evaluate import evaluate_model


def main(train_path, test_path, model_out, results_out):
    print('Training model...')
    clf, _ = train_model(train_path, model_out)
    print('Evaluating model...')
    res = evaluate_model(model_out, test_path, results_out)
    print('Done. Metrics saved to', results_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='kddtrain.csv')
    parser.add_argument('--test', default='kddtest.csv')
    parser.add_argument('--out-model', default='mini_ids_project/model.joblib')
    parser.add_argument('--out-results', default='mini_ids_project/results')
    args = parser.parse_args()

    main(args.train, args.test, args.out_model, args.out_results)
