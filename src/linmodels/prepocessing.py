# %%
from . import config

import re

import pandas as pd
from pandas.core.series import Series

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

config.load()

REMOVE_STRINGS = ("www.", "https://", "http://")
SPLIT_DOT = '.'
EMPTY_STR = ''

class NonPipelinePrepocessing():
    """Collection of prepocessing methods before pipeline
    """
    def split_X_y(self, df):
        y = df['prediction']
        mask = y == -1
        y.loc[mask] = 0
        X = df.drop(['id', 'prediction'], axis=1)        
        return X, y

    def split_url(self, df): 
        remove_function = lambda x: re.sub("|".join(REMOVE_STRINGS), "", x)       
        urls = df['url'].map(remove_function)
        
        urls = urls.str.split(SPLIT_DOT, n=1).values
        df[['url_start', 'url_end']] = pd.DataFrame(
            list(urls), index=df.index,            
        )
        return df

    def prepocess(self, df):
        _df= df.copy(deep=True)
        _df = self.split_url(_df)
        X, y = self.split_X_y(_df)
        return X, y

    def __call__(self, df):
        return self.prepocess(df)