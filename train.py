"""
训练主模型 BiLSTM-CRF
"""

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data_utils import (
    build_vocab,
    load_treebank_sentences,
    split_sentences,
    save_vocab,
)
from model import BiLSTMCRF
from train_utils import POSDataset, collate_batch, evaluate, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="训练 BiLSTM-CRF 主模型")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--plot-path",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "fig5-1_training_curves_crf.png"),
        help="训练曲线图路径",
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成曲线图")
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    sentences = load_treebank_sentences()
    train_s, val_s, test_s = split_sentences(sentences, config.RANDOM_SEED)
    word_to_idx, tag_to_idx, idx_to_tag = build_vocab(train_s, config.MIN_WORD_FREQ)
    save_vocab(word_to_idx, tag_to_idx, idx_to_tag, config.VOCAB_PATH)

    train_ds = POSDataset(train_s, word_to_idx, tag_to_idx)
    val_ds = POSDataset(val_s, word_to_idx, tag_to_idx)
    test_ds = POSDataset(test_s, word_to_idx, tag_to_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_batch)

    model = BiLSTMCRF(
        vocab_size=len(word_to_idx),
        num_tags=len(tag_to_idx),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        pad_idx=word_to_idx["<PAD>"],
        tag_pad_idx=tag_to_idx["<PAD>"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = config.MODEL_PATH

    best_val = 0.0
    history_epochs = []
    history_train_loss = []
    history_val_acc = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for word_pad, tag_pad, mask in bar:
            word_pad = word_pad.to(device)
            tag_pad = tag_pad.to(device)
            mask_d = mask.to(device)

            model.zero_grad()
            loss = model(word_pad, tag_pad, mask_d)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = evaluate(model, val_loader, device)
        mean_loss = total_loss / len(train_loader)
        history_epochs.append(epoch)
        history_train_loss.append(mean_loss)
        history_val_acc.append(val_acc)
        tqdm.write(f"epoch {epoch} loss={mean_loss:.4f} val_token_acc={val_acc:.4f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"model=BiLSTM-CRF  checkpoint={checkpoint_path}")
    print(f"best validation token accuracy (checkpoint): {best_val:.4f}")
    print(f"train token accuracy (best checkpoint): {train_acc:.4f}")
    print(f"test token accuracy: {test_acc:.4f}")

    if not args.no_plot and history_epochs:
        plot_training_curves(history_epochs, history_train_loss, history_val_acc, args.plot_path)
        print(f"training curve saved: {args.plot_path}")


if __name__ == "__main__":
    main()
