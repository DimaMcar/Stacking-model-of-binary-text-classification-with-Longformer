from . import config
from .prepocessing import NonPipelinePrepocessing

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class PipelineBuilder():
    """Pipeline builder class for the logistic regression model with grid search
    """
    def __init__(self):
        self.url_start = None,
        self.url_end = None,
        self.title_vectorizer = None,
        self.body_vectorizer = None
        self.preprocessor = None
        self.clf = None

    def get_split(self, train):
        X, y = NonPipelinePrepocessing()(train)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=0.2
        )        
        return X_train, X_test, y_train, y_test

    def url_transformers(self):
        self.url_start = CountVectorizer()
        self.url_end = CountVectorizer()                        
        return self.url_start, self.url_end

    def text_transformers(self):
        self.title_vectorizer = CountVectorizer()
        self.body_vectorizer = TfidfVectorizer()
        return self.title_vectorizer, self.body_vectorizer
    
    def get_preprocessor(self):
        url_start, url_end = self.url_transformers()
        title_vectorizer, body_vectorizer = self.text_transformers()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('url_start', url_start, 'url_start'),
                ('url_end', url_end, 'url_end'),
                ('title_vectorizer', title_vectorizer, 'textTitle'),
                ('body_vectorizer', body_vectorizer, 'textBody')
            ])
        return self.preprocessor
    
    def get_classifier(self):
        self.clf = LogisticRegression()
        return self.clf
    
    def get_pipeline(self):
        preprocessor = self.get_preprocessor()
        clf = self.get_classifier()

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('clf', clf)
        ])
        return pipeline