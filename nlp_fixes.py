"""
与 PTB 训练语料对齐的轻量预处理与标签启发式纠错。
Unicode 弯引号会导致 NLTK 将缩约拆成 [词, "'", "s]，引发错误标注；
部分缩约（No one's …）在训练中较少，易产生 CD/JJ 等误标。
常见输入问题：含中文的括号说明被误标、tobe 连写、短语动词 hand in 误标等。
"""

import re
from typing import List, Sequence

# 规范为 ASCII '，便于与 PTB tokenizer 对齐
_UNICODE_APOSTrophe = ("\u2018", "\u2019", "\u2032", "\u2035")

# 「… is …」常见缩约为 VBZ（与 PTB 中 's -> VBZ / POS 共存，语境下纠错）
_CONTRACT_IS_SUBJECT_LEMMAS = frozenset(
    (
        "there",
        "it",
        "that",
        "what",
        "who",
        "here",
        "where",
        "how",
        "when",
        "why",
        "everything",
        "nothing",
        "something",
    )
)

# model 若在缩约上给出明显不合理的标签，可被 VBZ 覆盖
_BAD_FOR_CONTRACT_IS = frozenset({"JJ", "RB", "IN", "CC", "TO", ".", ",", "POS"})

# 含中日韩等字符的圆括号说明（写作提示等），不作为英文词性标注对象
_RE_PAREN_WITH_CJK = re.compile(
    r"\([^)\n]{0,800}?[\u3000-\u30ff\u4e00-\u9fff\uac00-\ud7af][^)\n]{0,800}?\)"
    r"|（[^）\n]{0,800}?[\u3000-\u30ff\u4e00-\u9fff\uac00-\ud7af][^）\n]{0,800}?）",
    flags=re.UNICODE,
)

_RE_TOBE = re.compile(r"\btobe\b", re.IGNORECASE)
_RE_WS = re.compile(r"[ \t]+")

# 句号/问号/叹号后直接接字母或引号后的字母 → 避免 song.You
_RE_PUNCT_GLUE_WORD = re.compile(r"([.!?])([^\s\d.!?\)\]\}\"'])")

# 常见 OCR/粘贴误粘：tim eyou 类
_RE_FUSED_PRONOUN_TAIL = re.compile(
    r"\b(time|said|tell|told|heard|noticed|thought|everything|anything|something)(you|they|we)\b",
    re.IGNORECASE,
)
_RE_FUSED_YOUR_NOUN = re.compile(
    r"\b(your|our)(schoolwork|homework|mistake|thought|idea|book)\b",
    re.IGNORECASE,
)


def fix_glue_between_words(text: str) -> str:
    """减轻复制粘贴导致的缺空格。"""
    if not text:
        return text
    t = _RE_PUNCT_GLUE_WORD.sub(r"\1 \2", text)
    t = _RE_FUSED_PRONOUN_TAIL.sub(r"\1 \2", t)
    t = _RE_FUSED_YOUR_NOUN.sub(r"\1 \2", t)
    for wrong, fixed in (
        ("timeyou", "time you"),
        ("yourschoolwork", "your schoolwork"),
        ("everythingyou", "everything you"),
        ("anythingyou", "anything you"),
        ("somethingyou", "something you"),
    ):
        t = re.compile(re.escape(wrong), re.IGNORECASE).sub(fixed, t)
    return t

# 「to」后不定式原形；易被标成 NN
_INFINITIVE_VERBS_AFTER_TO = frozenset(
    {
        "be",
        "do",
        "go",
        "get",
        "make",
        "see",
        "come",
        "take",
        "give",
        "know",
        "think",
        "say",
        "tell",
        "use",
        "find",
        "work",
        "play",
        "practice",
        "try",
        "learn",
        "help",
        "call",
        "keep",
        "let",
        "put",
        "mean",
        "set",
        "show",
        "hear",
        "read",
        "write",
        "pay",
        "meet",
        "include",
        "continue",
        "move",
        "live",
        "believe",
        "bring",
        "happen",
    }
)


def _is_trace_like_tag(tag: str) -> bool:
    return bool(tag) and (tag.startswith("-NONE") or tag.startswith("*"))


def sanitize_punctuation_and_trace_tags(tokens: List[str], tags: List[str]) -> List[str]:
    """标点按字面 PTB；-NONE-/迹类落在真实词条上时用启发式替换。"""
    out = list(tags)
    n = len(tokens)
    for i in range(n):
        tok = tokens[i]

        if tok in {".", "!", "?"}:
            out[i] = "."
            continue
        if tok == ",":
            out[i] = ","
            continue
        if tok == ":":
            out[i] = ":"
            continue
        if tok == ";":
            out[i] = ":"
            continue
        if tok in {"``", "''"}:
            out[i] = tok
            continue
        if tok in {"-LRB-", "-RRB-"}:
            out[i] = tok
            continue
        if tok == "(":
            out[i] = "-LRB-"
            continue
        if tok == ")":
            out[i] = "-RRB-"
            continue

        lw = tok.lower()

        if _is_trace_like_tag(out[i]) and lw.isalpha():
            if lw.endswith("ful") or lw.endswith("less") or lw.endswith("ous") or lw in (
                "beautiful",
                "wonderful",
                "successful",
                "careful",
            ):
                out[i] = "JJ"
            elif lw.endswith("ly"):
                out[i] = "RB"
            elif lw.endswith("ing"):
                out[i] = "VBG"
            elif lw.endswith("ed"):
                out[i] = "JJ" if i == 0 or tokens[i - 1] == "," else "VBN"
            else:
                out[i] = "NN"

        if (
            lw == "beautiful"
            and i > 0
            and tokens[i - 1].lower() == "born"
            and out[i] not in {"JJ", "RB"}
        ):
            out[i] = "JJ"

    return out


def normalize_apostrophe(text: str) -> str:
    t = text
    for ch in _UNICODE_APOSTrophe:
        t = t.replace(ch, "'")
    return t


def strip_paren_cjk_commentary(text: str) -> str:
    """去掉括号内带中文/日/韩的说明性文字，避免进入分词与词性预测。"""
    return _RE_PAREN_WITH_CJK.sub(" ", text)


def normalize_for_tagging(text: str) -> str:
    """推理前统一清洗：撇号、去掉中文括号说明、常见连写修正。"""
    t = normalize_apostrophe((text or "").strip())
    t = strip_paren_cjk_commentary(t)
    t = fix_glue_between_words(t)
    t = _RE_TOBE.sub("to be", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def merge_split_contract_s(tokens: Sequence[str]) -> List[str]:
    """
    word + `'`/`'` + `s` -> word's（常见于弯引号经 normalize 后仍未合并的情况）。
    """
    apostrophe_toks = {"'", "`"}
    merged: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        if (
            i + 2 < n
            and tokens[i + 1] in apostrophe_toks
            and tokens[i + 2].lower() == "s"
            and tokens[i]
        ):
            merged.append(tokens[i] + "'s")
            i += 3
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def heuristic_fix_tags(tokens: List[str], tags: List[str]) -> List[str]:
    """在 BiLSTM-CRF 解码后做小范围 PTB 向纠错。"""
    if not tags or len(tags) != len(tokens):
        return tags
    out = sanitize_punctuation_and_trace_tags(tokens, list(tags))
    n = len(tokens)

    for i in range(n):
        lw = tokens[i].lower()

        # no one：one 在「没有人」义项下不应标为基数词 CD（PTB 常见 NN）
        if i > 0 and tokens[i - 1].lower() == "no" and lw == "one" and out[i] == "CD":
            out[i] = "NN"

        # 「is born / 's born」：born 为过去分词 VBN
        if lw == "born" and i > 0 and tokens[i - 1] == "'s":
            out[i] = "VBN"

        if tokens[i] != "'s":
            continue

        # No one's (= No one is) -> 's -> VBZ
        if (
            i >= 2
            and tokens[i - 2].lower() == "no"
            and tokens[i - 1].lower() == "one"
        ):
            out[i] = "VBZ"
            continue

        prev = tokens[i - 1].lower() if i > 0 else ""
        if prev in _CONTRACT_IS_SUBJECT_LEMMAS and out[i] in _BAD_FOR_CONTRACT_IS | {"NN"}:
            out[i] = "VBZ"

    # 短语、不定式、复合词等非 's 相关修正
    for i in range(n):
        lw = tokens[i].lower()
        if lw == "fragmented" and out[i] == "NNP":
            out[i] = "JJ"
        if lw == "schoolwork" and out[i] == "NNS":
            out[i] = "NN"

    for i in range(1, n):
        lw = tokens[i].lower()
        if tokens[i - 1].lower() == "to" and lw in _INFINITIVE_VERBS_AFTER_TO:
            if out[i] == "NN":
                out[i] = "VB"

    for i in range(1, n):
        lw = tokens[i].lower()
        prev = tokens[i - 1].lower()
        if prev == "hand" and lw == "in" and out[i] in ("NN", "IN"):
            out[i] = "RP"

    for i in range(n - 1):
        if (
            tokens[i].lower() == "hand"
            and tokens[i + 1].lower() == "in"
            and out[i + 1] == "RP"
            and out[i] == "NN"
        ):
            out[i] = "VB"

    for i in range(1, n):
        lw = tokens[i].lower()
        prev = tokens[i - 1].lower()
        if prev in ("digital", "online") and lw == "reading" and out[i] == "JJ":
            out[i] = "NN"

    return out

