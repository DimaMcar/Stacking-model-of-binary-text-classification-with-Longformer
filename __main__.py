import config

import pandas as pd
from src.predict import train_predict

train = pd.read_csv(config.TRAIN_PATH, encoding='latin')
test = pd.read_csv(config.TEST_PATH, encoding='latin')

predict = train_predict(train, test)
predict[['id', 'prediction']].to_csv('submission.csv', index=False)