import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent.parent)
DATA_PATH = ROOT_PATH + "/data/"
TRAIN_PATH = DATA_PATH + "train.csv"
MODEL_PATH = ROOT_PATH + "/models/"

PARAM_GRID = {
    'preprocessor__url_start__stop_words': [None, 'english'],
    'preprocessor__url_start__max_features': [100, 500, 1000, 2000, 3000],
    'preprocessor__url_start__min_df': [1, 0.1, 0.2],
    'preprocessor__url_start__max_df': [1.0, 0.9, 0.8],

    'preprocessor__url_end__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'preprocessor__url_end__stop_words': [None, 'english'],
    'preprocessor__url_end__max_features': [500, 1000, 2000, 3000, 5000, 10000, 20000],
    'preprocessor__url_end__binary': [True, False],
    'preprocessor__url_end__min_df': [1, 0.1, 0.2],
    'preprocessor__url_end__max_df': [1.0, 0.9, 0.8],

    'preprocessor__title_vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
    'preprocessor__title_vectorizer__stop_words': [None, 'english'],
    'preprocessor__title_vectorizer__max_features': [1000, 2000, 3000, 5000, 10000, 20000, 30000],
    'preprocessor__title_vectorizer__binary': [True, False],
    'preprocessor__title_vectorizer__min_df': [1, 0.1, 0.2],
    'preprocessor__title_vectorizer__max_df': [1.0, 0.9, 0.8],

    'preprocessor__body_vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)],
    'preprocessor__body_vectorizer__stop_words': [None, 'english'],
    'preprocessor__body_vectorizer__max_features': [3000, 5000, 10000, 20000, 30000, 50000, 70000],
    'preprocessor__body_vectorizer__binary': [True, False],    
    'preprocessor__body_vectorizer__min_df': [1, 0.1, 0.2],
    'preprocessor__body_vectorizer__max_df': [1.0, 0.9, 0.8],
    'preprocessor__body_vectorizer__norm': ['l1', 'l2'],
    'preprocessor__body_vectorizer__use_idf': [True, False],
    'preprocessor__body_vectorizer__sublinear_tf': [True, False],

    'clf__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'clf__dual': [True, False],
    'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'clf__C': [0.6, 0.8, 1.0, 1.2, 1.4],
    'clf__intercept_scaling': [0.8, 1, 1.2],
    'clf__solver': ['liblinear', 'newton-cg', 'saga', 'sag', 'lbfgs'],
    'clf__l1_ratio': [0.5, 1, 0],  
}

def load():
    sys.path.insert(0, ROOT_PATH)