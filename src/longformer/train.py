from . import config
from .dataset import TextDataset
from .model import LongformerClass
from .engine import inference_fn, train_fn, valid_fn

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import warnings

from torch.utils.data import DataLoader
from torch import nn
import torch

import numpy as np
import random
import os

warnings.filterwarnings("ignore")


class Training:
    """Longformer class for training
    """
    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def train(self, df=None):
        dataset = TextDataset(df)
        dataloader = DataLoader(dataset)

        skf = StratifiedKFold(n_splits=config.N_SPLITS,
                              shuffle=True,
                              random_state=42)
        X = dataset.text
        y = dataset.target

        for i, (train_index, val_index) in enumerate(skf.split(X, y)):
            if os.path.exists(config.MODEL_PATH + "model_{i}.pth"):
                continue
            self.seed_everything(i)
            torch.cuda.empty_cache()
            model = LongformerClass()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(config.DEVICE)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=config.LEARNING_RATE
            )

            train_fold = df.loc[train_index, :].reset_index(drop=True)
            val_fold = df.loc[val_index, :].reset_index(drop=True)

            train = TextDataset(train_fold)
            train_loader = DataLoader(train, config.BATCH_SIZE, shuffle=True)

            val = TextDataset(val_fold)
            val_loader = DataLoader(val, config.BATCH_SIZE, shuffle=False)

            best_loss = 0
            best_preds = None

            epoch_params = {
                'model': model,
                'optimizer': optimizer,
                'train_loader': train_loader,
                'val': val,
                'val_loader': val_loader,
                'best_loss': best_loss,
                'best_preds': best_preds,
                'i': i
            }

            for epoch in range(config.EPOCHS):
                print(f"EPOCH: {epoch}")
                self.train_epoch(**epoch_params)
            df.loc[val_index, 'probs'] = best_preds[:, 1]
            df.to_csv(config.DATA_PATH + "CV_predics.csv", index=False)
        df = pd.read_csv(config.DATA_PATH + "CV_predics.csv")
        return df['probs']

    def train_epoch(self, model, optimizer, train_loader, val_loader, val, best_loss, i, best_preds):
        train_loss = train_fn(
            model, optimizer, train_loader, config.DEVICE
        )
        print(f"train_loss: {train_loss}")

        valid_loss, valid_preds = valid_fn(
            model, val_loader, config.DEVICE
        )

        print(f"valid_loss: {valid_loss}")

        preds = np.argmax(valid_preds, axis=1)
        acc = accuracy_score(preds, val.target)
        print(f'model accuracy: {acc}')
        if acc > best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH + "model_{i}.pth")
            best_loss = acc
            best_preds = valid_preds

            print(f'best_model!!!, accuracy: {acc}')

    def stack_predict(self, train, test):
        self.train(train)
        return self.predict_full(test)

    def predict_full(self, data):
        test = TextDataset(data)
        test_loader = DataLoader(test, config.BATCH_SIZE, shuffle=False)
        final_preds = []
        for i in range(config.N_SPLITS):
            torch.cuda.empty_cache()
            model = LongformerClass()
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(f"model_{i}.pth"))
            model.to(config.DEVICE)
            preds = inference_fn(model, test_loader, config.DEVICE)
            final_preds.append(preds.reshape((-1, 1)))
        final_preds = np.concatenate(final_preds, axis=1)
        return final_preds.apply(np.mean, axis=1)
