import torch


def yield_tokens(data, tokenizer):
    """Generator, which goes through data and applies tokenizer to each sentence

    Args:
        data (iterable): list or other iterable structure with text
        tokenizer (Callable): tokenization function

    Yields:
        iterable: token sequence for each sentence
    """
    for sentence in data:
        yield tokenizer(sentence)


def collate_batch(batch, vocab):
    """collate function for DataLoader, which collects data into batch and pads it

    Args:
        batch (list): List of tuples (text, label), returned by Dataset
        vocab (Vocab): torchtext token vocabulary with indices

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
