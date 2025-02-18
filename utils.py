import torch
from torch.utils.data import DataLoader, random_split


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
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
