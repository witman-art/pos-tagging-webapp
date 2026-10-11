import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pad_idx: int = 0,
        tag_pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.num_tags = num_tags
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, words, tags, mask):
        """
        words: [B, T] long
        tags: [B, T] long (use tag_pad_idx for padding)
        mask: [B, T] bool  True 表示有效 token
        """
        x = self.dropout(self.embed(words))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.fc(x)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return loss

    def decode(self, words, mask):
        x = self.dropout(self.embed(words))
        x, _ = self.lstm(x)
        emissions = self.fc(x)
        return self.crf.decode(emissions, mask=mask)
