import json
import random
from collections import Counter
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import treebank
from nltk.tokenize import word_tokenize

from nlp_fixes import merge_split_contract_s, normalize_for_tagging

PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_TAG = "<PAD>"


def augment_pos_training_sents():
    """
    增补少量与 PTB 一致的高频缩约 / 固定搭配句，减轻「No one's」等 OOV 语境误标。
    随主语料一同参与划分与训练。
    """
    return [
        [
            ("No", "DT"),
            ("one", "NN"),
            ("'s", "VBZ"),
            ("born", "VBN"),
            ("being", "VBG"),
            ("good", "JJ"),
            ("at", "IN"),
            ("all", "DT"),
            ("things", "NNS"),
            (".", "."),
        ],
        [
            ("There", "EX"),
            ("'s", "VBZ"),
            ("no", "DT"),
            ("problem", "NN"),
            (".", "."),
        ],
        [
            ("It", "PRP"),
            ("'s", "VBZ"),
            ("not", "RB"),
            ("a", "DT"),
            ("problem", "NN"),
            (".", "."),
        ],
    ]


def ensure_nltk():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def load_treebank_sentences() -> List[List[Tuple[str, str]]]:
    ensure_nltk()
    sents = [list(s) for s in treebank.tagged_sents()]
    sents.extend(augment_pos_training_sents())
    return sents


def build_vocab(sentences: List[List[Tuple[str, str]]], min_freq: int):
    words = Counter()
    tags = set()
    for sent in sentences:
        for w, t in sent:
            words[w.lower()] += 1
            tags.add(t)
    word_list = [w for w, c in words.items() if c >= min_freq]
    word_to_idx: Dict[str, int] = {PAD_WORD: 0, UNK_WORD: 1}
    for i, w in enumerate(sorted(word_list), start=2):
        word_to_idx[w] = i
    tag_to_idx: Dict[str, int] = {PAD_TAG: 0}
    for i, t in enumerate(sorted(tags), start=1):
        tag_to_idx[t] = i
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    return word_to_idx, tag_to_idx, idx_to_tag


def sent_to_indices(
    sent: List[Tuple[str, str]],
    word_to_idx: Dict[str, int],
    tag_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int]]:
    w_ids = [word_to_idx.get(w.lower(), word_to_idx[UNK_WORD]) for w, _ in sent]
    t_ids = [tag_to_idx[t] for _, t in sent]
    return w_ids, t_ids


def split_sentences(
    sentences: List[List[Tuple[str, str]]],
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[list, list, list]:
    rng = random.Random(seed)
    data = sentences[:]
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def simple_tokenize(text: str) -> List[str]:
    text = normalize_for_tagging(text)
    if not text:
        return []
    return merge_split_contract_s(word_tokenize(text))


def save_vocab(word_to_idx: Dict, tag_to_idx: Dict, idx_to_tag: Dict, path: str):
    # JSON keys must be str; idx_to_tag keys are ints
    payload = {
        "word_to_idx": word_to_idx,
        "tag_to_idx": tag_to_idx,
        "idx_to_tag": {str(k): v for k, v in idx_to_tag.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    word_to_idx = payload["word_to_idx"]
    tag_to_idx = payload["tag_to_idx"]
    idx_to_tag = {int(k): v for k, v in payload["idx_to_tag"].items()}
    return word_to_idx, tag_to_idx, idx_to_tag
