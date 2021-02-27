from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from . import config

class LongformerClass(torch.nn.Module):
    """Longformer model class that replicates original longformer, but is necessary for paralellism

    Args:
        torch ([type]): [description]
    """
    def __init__(self):
        super(LongformerClass, self).__init__()
        self.base_model = LongformerForSequenceClassification.from_pretrained(config.BASE_MODEL, 
                                                                              config=config.MODEL_CONFIG)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return x