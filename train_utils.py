"""训练脚本共用：数据集、评测、曲线图（主模型与对比实验复用）。"""

import os

import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data_utils import sent_to_indices


class POSDataset(Dataset):
    def __init__(self, sentences, word_to_idx, tag_to_idx):
        self.samples = []
        for s in sentences:
            w_ids, t_ids = sent_to_indices(s, word_to_idx, tag_to_idx)
            if not w_ids:
                continue
            self.samples.append((torch.tensor(w_ids, dtype=torch.long), torch.tensor(t_ids, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate_batch(batch):
    words, tags = zip(*batch)
    word_pad = pad_sequence(words, batch_first=True, padding_value=0)
    tag_pad = pad_sequence(tags, batch_first=True, padding_value=0)
    mask = word_pad != 0
    return word_pad, tag_pad, mask


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for word_pad, tag_pad, mask in loader:
            word_pad = word_pad.to(device)
            mask_d = mask.to(device)
            pred_sequences = model.decode(word_pad, mask_d)
            for i, pred in enumerate(pred_sequences):
                m = mask[i]
                length = m.sum().item()
                gold = tag_pad[i, :length].tolist()
                correct += sum(int(p == g) for p, g in zip(pred, gold))
                total += length
    model.train()
    return correct / max(total, 1)


def plot_training_curves(epochs: list, train_losses: list, val_accs: list, out_path: str) -> None:
    if len(epochs) != len(train_losses) or len(epochs) != len(val_accs):
        raise ValueError("epochs / train_losses / val_accs 长度必须一致")

    fig, ax_loss = plt.subplots(figsize=(8, 4.5))
    color_loss = "#1976d2"
    color_acc = "#c62828"

    (line_loss,) = ax_loss.plot(
        epochs,
        train_losses,
        color=color_loss,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="Training loss (mean)",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss", color=color_loss)
    ax_loss.tick_params(axis="y", labelcolor=color_loss)
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, linestyle=":", alpha=0.5)

    ax_acc = ax_loss.twinx()
    (line_acc,) = ax_acc.plot(
        epochs,
        val_accs,
        color=color_acc,
        marker="s",
        markersize=4,
        linewidth=1.8,
        label="Validation token accuracy",
    )
    ax_acc.set_ylabel("Validation token accuracy", color=color_acc)
    ax_acc.tick_params(axis="y", labelcolor=color_acc)
    ax_acc.set_ylim(0.0, 1.0)

    handles = [line_loss, line_acc]
    labels = [h.get_label() for h in handles]
    ax_loss.legend(handles, labels, loc="center right", framealpha=0.92)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
