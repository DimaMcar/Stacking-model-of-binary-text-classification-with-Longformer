from . import config
from .pipeline import PipelineBuilder
from .prepocessing import NonPipelinePrepocessing

import os
import joblib

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import pandas as pd
import numpy as np

class ModelFitter:
    """Training class for the pipeline
    """
    def __init__(self):
        self.pipeline = PipelineBuilder().get_pipeline()
        self.X = None
        self.y = None
        self.grid = None
        self.model = None

    def set_dataset(self, train=None):
        if train is None:
            train = pd.read_csv(config.TRAIN_PATH,
                encoding='latin'
            )
        self.X, self.y = NonPipelinePrepocessing()(train)
        return self.X, self.y

    def set_grid(self, grid_param):
        self.grid = RandomizedSearchCV(
            self.pipeline,
            grid_param,
            cv=10,
            n_jobs=-1,
            error_score=np.nan,
            refit=False,
            n_iter=20000
        )
        return self.grid
    
    def fit_grid(self, grid_param, train=None):
        if train is not None:
            X, y = self.set_dataset(train)
        X, y = self.X, self.y
        if self.grid is None:
            self.grid = self.set_grid(grid_param)
        self.grid.fit(X, y)
        joblib.dump(self.grid, 'grid.bin')
        return self.grid

    
    def stack_predict(self, train=None, grid_param=None):
        if os.path.exists(config.MODEL_PATH+'lin_model.bin'):
            self.model = joblib.load(config.MODEL_PATH+'lin_model.bin')
            self.grid =   joblib.load(config.MODEL_PATH+'grid.bin')          
        if self.grid is None:
            grid = self.fit_grid(train, grid_param) 
            joblib.dump(grid, config.MODEL_PATH+'grid.bin')                           
        grid = self.grid
        self.model = grid.best_estimator_
        
        X, y = self.X, self.y
        skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

        for train_index, val_index in skf.split(X, y): 
            self.model.fit(
                X.loc[train_index, :],
                y.loc[train_index],                
            )
            probs = self.model.predict_proba(X.loc[val_index, :])
            X.loc[val_index, 'predict'] = probs[:, 1]
        self.model.fit(X, y)
        return X['predict'].values