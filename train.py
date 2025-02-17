import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.transforms import VocabTransform
from torchtext.vocab import build_vocab_from_iterator

from dataset import EmotionDataset
from model import EmotionClassifier
from utils import collate_batch, yield_tokens

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_DIR = "data/combined_emotion.csv"

df = pd.read_csv(DATASET_DIR)
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(
    yield_tokens(df, tokenizer), specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])

dataset = EmotionDataset(df, vocab, tokenizer)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_batch)

for batch in dataloader:
    texts, labels, lengths = batch
    print(texts.shape, labels.shape, lengths.shape)
    break
