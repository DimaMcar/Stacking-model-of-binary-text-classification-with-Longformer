from . import config

import torch

import pandas as pd

config.load()

class TextDataset:
    """Dataset class following PyTorch interface
    """
    def __init__(self, df: pd.DataFrame):
        self.text, self.target = self.preprocess(df)
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def preprocess(self, df):
        text = df['url'].map(str) +' '+ df['textTitle'] +' '+df['textBody']
        text = text.apply(lambda row: row.strip())
        target = df['prediction'].copy(deep=True)
        mask = target == -1
        target.loc[mask] = 0
        return text, target
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.tokenizer.encode_plus(
            self.text[item],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.target[item], dtype=torch.long)
        }