# import torch.nn as nn

import pandas as pd
import pytorch_lightning as pl
from transformers import BertTokenizer
import re
import pickle

class PreprocessorToxic():
    def load_data(self):
        data = pd.read_csv('data/preprocessed_indonesian_toxic_tweet.csv')
        data = data[data['Tweet'].notna()]
        tweet = data["Tweet"].values.tolist()  # type: ignore
        label = data.drop(["Tweet"], axis = 1)
        label = label.values.tolist()
        print(label)

if __name__ == '__main__':
    pretox = PreprocessorToxic()
    pretox.load_data()