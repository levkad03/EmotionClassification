import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader

from dataset import EmotionDataset
from model import EmotionClassifier
from utils import collate_batch, yield_tokens
from vocabulary import Vocabulary

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_DIR = "data/combined_emotion.csv"

df = pd.read_csv(DATASET_DIR)
vocab = Vocabulary()
for sentence in df["sentence"]:
    vocab.add_sentence(sentence)

dataset = EmotionDataset(df, vocab)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)

for batch in dataloader:
    texts, labels, lengths = batch
    print(texts.shape, labels.shape, lengths.shape)
    break
