import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATASET_DIR,
    DEVICE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    LEARNING_RATE,
    LOAD_MODEL,
    NUM_EPOCHS,
    NUM_LAYERS,
    NUM_WORKERS,
    PAD_IDX,
    PIN_MEMORY,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from dataset import EmotionDataset
from model import EmotionClassifier
from utils import (
    calculate_accuracy,
    create_dataloaders,
    load_checkpoint,
    save_checkpoint,
)
from vocabulary import Vocabulary

torch.manual_seed(RANDOM_SEED)


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, writer):
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (texts, labels, lengths) in enumerate(loop):
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.amp.autocast(DEVICE):
            outputs = model(texts, lengths)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item())

        writer.add_scalar(
            "Train/Batch_Loss", loss.item(), epoch * len(loader) + batch_idx
        )

    train_loss = total_loss / len(loader)
    train_accuracy = correct / total

    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_accuracy, epoch)

    return train_loss, train_accuracy


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
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    writer = SummaryWriter("runs/emotion_classifier_bidirectional")

    dummy_texts = torch.randint(0, len(vocab), (1, 10)).to(DEVICE)
    dummy_lengths = torch.tensor([10])
    writer.add_graph(model, (dummy_texts, dummy_lengths))

    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, epoch, writer
        )
        test_accuracy = calculate_accuracy(test_loader, model, device=DEVICE)

        writer.add_scalar("Test/Accuracy", test_accuracy, epoch)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
