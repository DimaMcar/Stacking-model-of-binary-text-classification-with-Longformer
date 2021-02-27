import joblib
from . import config
from .linmodels.train import ModelFitter
from .longformer.train import Training

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression


class Stack:
    """Ensemble stacking model of linear pipeline model and longformer model
    """

    def __init__(self):
        self.longformer_train = Training()
        self.lin_train = ModelFitter()
        self.lin_train.set_dataset()
        self.y = self.lin_train.y
        self.X = None
        self.model = None
        self.lin_model = None

    def get_predicts(self, train, test):
        long_preds = self.longformer_train.stack_predict(train, test)
        lin_preds = self.lin_train.stack_predict(train)
        self.lin_model = self.lin_train.model
        return np.concatenate([
            long_preds.reshape((-1, 1)),
            lin_preds.reshape((-1, 1))],
            axis=1
        )

    def train_stack(self, train, test):
        self.X = self.get_predicts(train, test)

        clf = LogisticRegression()
        score = cross_val_score(clf, self.X, self.y, cv=10, n_jobs=-1)
        print(f"Stacking logistic score: {np.mean(score)}")

        clf.fit(self.X, self.y)
        self.model = clf
        joblib.dump(clf, config.STACK_MODEL_PATH)

        self.lin_model = self.lin_train.model
        self.lin_model.fit(self.lin_train.X, self.lin_train.y)

        return clf
