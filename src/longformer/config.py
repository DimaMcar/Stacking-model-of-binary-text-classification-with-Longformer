import sys
from pathlib import Path

from transformers import LongformerTokenizer, LongformerForSequenceClassification


ROOT_PATH = str(Path(__file__).parent.parent.parent)
DATA_PATH = ROOT_PATH + "/data/"
TRAIN_PATH = DATA_PATH + "train.csv"
MAX_LEN = 4096
TOKENIZER = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
DEVICE = 'cpu'
BASE_MODEL = 'allenai/longformer-base-4096'
MODEL_PATH = ROOT_PATH + "/models/"
EPOCHS = 4
LEARNING_RATE = 1e-05
N_SPLITS = 5
BATCH_SIZE = 4
DROPOUT = 0.2

MODEL_CONFIG = LongformerForSequenceClassification.from_pretrained(
        BASE_MODEL,
        return_dict=True).config

MODEL_CONFIG.hidden_dropout_prob = DROPOUT 

def load():
    sys.path.insert(0, ROOT_PATH)