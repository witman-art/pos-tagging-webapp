"""
对比实验专用模型（无 CRF）：不用于 Web 推理与主系统。

BiLSTM + Linear 发射分数，逐位置 CrossEntropyLoss，预测时 argmax。
与 BiLSTMCRF 保持相同的 embedding / BiLSTM / 线性层维度配置，便于公平对比。
"""

import torch
import torch.nn as nn


class BiLSTMTagger(nn.Module):
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
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tag_pad_idx, reduction="mean")

    def forward(self, words, tags, mask):
        """
        words: [B, T] long
        tags: [B, T] long（填充位为 tag_pad_idx）
        mask: [B, T] bool，与 BiLSTMCRF 接口一致
        """
        _ = mask
        x = self.dropout(self.embed(words))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.fc(x)
        emissions_flat = emissions.reshape(-1, self.num_tags)
        tags_flat = tags.reshape(-1)
        return self.ce_loss(emissions_flat, tags_flat)

    def decode(self, words, mask):
        x = self.dropout(self.embed(words))
        x, _ = self.lstm(x)
        emissions = self.fc(x)
        pred = emissions.argmax(dim=-1)
        sequences = []
        for i in range(pred.size(0)):
            length = mask[i].sum().item()
            sequences.append(pred[i, :length].tolist())
        return sequences


# 论文叙述中的「纯 BiLSTM 基线」
BiLSTM = BiLSTMTagger
