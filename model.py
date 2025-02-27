import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        pad_idx,
        num_layers,
        dropout=0.3,
    ):
        super(EmotionClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )  # bidirectional LSTM to capture both directions

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Для bidirectional LSTM нужно объединить два направления
        hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )  # (batch_size, hidden_dim * 2)

        output = self.fc(self.dropout(hidden))
        return output
