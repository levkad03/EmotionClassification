import torch
from torch.utils.data import DataLoader, random_split


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves model checkpoint

    Args:
        state : State of the model
        filename (str, optional): Name of the file. Defaults to "my_checkpoint.pth.tar".
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """Load model checkpoint from the file

    Args:
        checkpoint: model checkpoint
        model: model which checkpoint will be loaded
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def collate_batch(batch):
    """collate function for DataLoader, which collects data into batch and pads it

    Args:
        batch (list): List of tuples (text, label), returned by Dataset

    Returns:
        tuple:
            - texts (Tensor): padded tensor
            - lengths (Tensor): Text lengths before padding
            - labels (Tensor): tensor with labels
    """
    texts, labels = zip(*batch)
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    return texts, labels, lengths


def create_dataloaders(
    dataset, batch_size, num_workers, pin_memory, train_split, random_seed=123
):
    """Create train and test dataloaders from dataset

    Args:
        dataset : dataset which will be loaded
        batch_size (int): batch size
        num_workers (int): number of workers for multiprocessing
        pin_memory (bool): pin memory
        train_split (float): split ration
        random_seed (int, optional): random seed. Defaults to 123.

    Returns:
        tuple: train and test dataloaders
    """
    torch.manual_seed(random_seed)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def calculate_accuracy(loader, model, device):
    """Calculate accuracy of the model

    Args:
        loader (DataLoader): dataloader of the dataset
        model (Module): Neural network
        device (string): device which will be used

    Returns:
        float: accuracy of the model
    """
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts, lengths)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total
