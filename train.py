import pandas as pd
import torch
from tqdm import tqdm

from dataset import EmotionDataset
from model import EmotionClassifier
from utils import (
    create_dataloaders,
    load_checkpoint,
    save_checkpoint,
)
from vocabulary import Vocabulary

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DATASET_DIR = "data/combined_emotion.csv"
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
PAD_IDX = 0
NUM_LAYERS = 2
RANDOM_SEED = 123
WEIGHT_DECAY = 0

torch.manual_seed(RANDOM_SEED)

print(f"Device: {DEVICE}")


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader, desc=f"Epoch {epoch}")

    for texts, labels, lengths in loop:
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast(DEVICE):
            outputs = model(texts, lengths)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    df = pd.read_csv(DATASET_DIR)
    vocab = Vocabulary()
    for sentence in df["sentence"]:
        vocab.add_sentence(sentence)

    dataset = EmotionDataset(df, vocab)

    train_loader, test_loader = create_dataloaders(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        train_split=0.8,
        random_seed=RANDOM_SEED,
    )

    model = EmotionClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(dataset.labels),
        pad_idx=PAD_IDX,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("emotion_classifier.pth.tar"), model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "emotion_classifier.pth.tar")


if __name__ == "__main__":
    main()
