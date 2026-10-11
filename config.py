"""训练与推理的默认超参数（可按显卡与篇幅调整）。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

EMBEDDING_DIM = 100
HIDDEN_DIM = 200
MIN_WORD_FREQ = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.005
MAX_EPOCHS = 15
DROPOUT = 0.5
CLIP_GRAD = 5.0
RANDOM_SEED = 42

# 默认保存的权重与词典（训练后生成）
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pos_bilstm_crf.pt")
# 无 CRF 基线（BiLSTM + 逐位置 CE），与完整模型共用词汇表文件
BASELINE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pos_bilstm_ce.pt")
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "vocab.json")
