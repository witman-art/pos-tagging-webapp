import re
from typing import List, Tuple

import torch
import nltk
from nltk.tokenize import word_tokenize

import config
from data_utils import load_vocab, PAD_WORD
from model import BiLSTMCRF
from nlp_fixes import heuristic_fix_tags, merge_split_contract_s, normalize_for_tagging


def safe_sent_tokenize(text: str) -> List[str]:
    """Prefer NLTK sentence splitting; fall back to regex when local data is missing."""
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]


def safe_word_tokenize(text: str) -> List[str]:
    """Prefer NLTK tokenization; fall back to a simple PTB-like tokenizer."""
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]", text)


class POSTagger:
    def __init__(self, model_path: str = None, vocab_path: str = None, device: str = None):
        self.model_path = model_path or config.MODEL_PATH
        self.vocab_path = vocab_path or config.VOCAB_PATH
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.word_to_idx, self.tag_to_idx, self.idx_to_tag = load_vocab(self.vocab_path)
        self.model = BiLSTMCRF(
            vocab_size=len(self.word_to_idx),
            num_tags=len(self.tag_to_idx),
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=0.0,
            pad_idx=self.word_to_idx[PAD_WORD],
            tag_pad_idx=self.tag_to_idx["<PAD>"],
        )
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._ensure_tokenizer()

    def _ensure_tokenizer(self):
        self.nltk_tokenizer_ready = True
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(resource)
            except LookupError:
                self.nltk_tokenizer_ready = False

    def tokenize(self, text: str) -> List[str]:
        text = normalize_for_tagging(text)
        if not text:
            return []
        tokens = safe_word_tokenize(text)
        return merge_split_contract_s(tokens)

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
            return []
        ids = [
            self.word_to_idx.get(t.lower(), self.word_to_idx["<UNK>"])
            for t in tokens
        ]
        word_t = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones(1, len(ids), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            seq = self.model.decode(word_t, mask)[0]
        tags = [self.idx_to_tag[i] for i in seq]
        tags = heuristic_fix_tags(tokens, tags)
        return list(zip(tokens, tags))

    def tag_text(self, text: str) -> List[Tuple[str, str]]:
        return self.tag_tokens(self.tokenize(text))


def split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tag_text_batch(tagger: POSTagger, text: str) -> List[List[Tuple[str, str]]]:
    """按空行分段，每段内按句标注（句级用于展示）。"""
    text = normalize_for_tagging(text)
    results = []
    for para in split_paragraphs(text):
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            sents = safe_sent_tokenize(line)
            for s in sents:
                tagged = tagger.tag_text(s)
                if tagged:
                    results.append(tagged)
    if not results and text.strip():
        results.append(tagger.tag_text(text))
    return results
