# Stacking-model-of-binary-text-classification-with-Longformer

## Requirments
ubuntu==18.04
python==3.7.9
pandas==1.1.3
transformers==3.4.0
torch==1.3.1
numpy==1.19.2
joblib==0.17.0
scikit_learn==0.23.2
## Description
This is a stack ensemble model for predicting relevance of an article for a sigle customer. It consists of 2 models:
  - Logistic regression model
  - Longformer model
You can run train and prediction with 'python DStask'
## Logistic regression
For logistic regression model preprocessing was made to remove from 'url' field repeating, non-informative strings and stplitting it into two fields. Further preprocessing was build into the pipeline to perform Random Search of hyperpameters. Since dataset isn't large, it's reasonable to try simpler models first. CountVectorizer was used for all the columns, except for the textBody. TextBody is large enough string to ecode it as tf-ids, which is more informative. 
## Longformer
Longformer model was used with preprocessing that combined all the text into a single sequence. Longformer pretrained model was used. This model is escpesially effective on long documents up with window lenth up to 4096 tokens. It was trained in paralell on 4 gpus V100 in Ubuntu 18.04. Additional regularization of Dropout = 0.2 was used, because the dataset isn't large enough to prevent overfitting. Cross-validation of 5-folds was used. So, 5 models in total was trained with different seeds and their prediction was used to stacking. The final prediction for test set was made with the average of all 5 predictors. 
## Stacking
Stacking combined predictions of two models of validation data and simple logistic regression was used to predict the final class. The stacking model with additional features as text was tried but showed worse result. We assume the data in the test set is distributed the same way as in train dataset as 2:1. 
