from .stacking import Stack
from . import config

import numpy as np
import pandas as pd


def train_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Train and get predicts from the test set for the final stack model

    Args:
        train (pd.DataFrame): input train dataset
        test (pd.DataFrame): input test dataset

    Returns:
        pd.DataFrame: predicted test set
    """
    stack = Stack()
    stack_model = stack.train_stack(train, test)
    lin_model = stack.lin_model
    test.loc[:, 'prediction'] = 0
    X_lin, _ = stack.lin_train.set_dataset(test)
    X_lin.loc[:, 'predict'] = 0
    lin_preds = lin_model.predict_proba(X_lin)[:, 1]
    long_preds = pd.read_csv('preds.csv')
    x = np.concatenate([
        long_preds['predictions'].values.reshape((-1, 1)),
        lin_preds.reshape((-1, 1))],
        axis=1
    )
    test['prediction'] = np.argmax(stack_model.predict_proba(x), axis=1)
    return test
