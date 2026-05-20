import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import jieba
jieba.setLogLevel(jieba.logging.WARN)  # 关闭jieba日志，加速
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('dl_baseline_visualizations_jieba', exist_ok=True)

# ===================== 配置项 =====================
class Config:
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"
    SOLUTION_PATH = "question_with_knowledge_solutions.json"
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    N_FOLDS = 5
    BATCH_SIZE = 32     #16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = ['BiLSTM-Attention', 'TextCNN', 'BiLSTM-CNN']

# ===================== 数据加载 =====================
def load_annotation_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"标注文件不存在：{path}")
    try:
        df = pd.read_csv(path, encoding='utf-8', sep=',', quotechar='"', escapechar='\\')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='gbk', sep=',', quotechar='"', escapechar='\\')
    df.columns = [col.strip() for col in df.columns]
    required_cols = ['问题id', '知识点名称', '认知目标层次', '知识点类型']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"需包含：{required_cols}")
    df = df.drop_duplicates(subset=['问题id', '知识点名称'], keep='first')
    return df

def load_solution_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"解题步骤文件不存在：{path}")
    with open(path, 'r', encoding='utf-8') as f:
        solution_json = json.load(f)
    solution_df = pd.DataFrame(solution_json['data_list'])[[
        'question_id', 'solution_steps', 'knowledge_points'
    ]].rename(columns={'question_id': '问题id', 'knowledge_points': '知识点列表'})
    solution_df = solution_df.drop_duplicates(subset=['问题id'], keep='first')
    return solution_df

def reconstruct_data_for_kv_retrieval(annotation_df: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
    annotation_df['问题id'] = annotation_df['问题id'].astype(str).str.strip()
    annotation_df['知识点名称'] = annotation_df['知识点名称'].astype(str).str.strip()
    solution_df['问题id'] = solution_df['问题id'].astype(str).str.strip()
    merged_df = pd.merge(annotation_df, solution_df[['问题id', 'solution_steps']], on='问题id', how='inner')
    merged_df = merged_df.rename(columns={'solution_steps': '解题步骤'})
    merged_df = merged_df[(merged_df['解题步骤'].notna()) & (merged_df['认知目标层次'].notna())]
    processed_data = []
    for qid, group in merged_df.groupby('问题id'):
        text = group['解题步骤'].iloc[0]
        for _, row in group.iterrows():
            kp = row['知识点名称']
            level = row['认知目标层次']
            processed_data.append({
                'question_id': qid,
                'fusion_text': f"解题过程:{text}；知识点:{kp}",
                'value': level
            })
    return pd.DataFrame(processed_data)

# ===================== 预分词（核心：只分一次，不卡死） =====================
def pre_segment_all_texts(data_df):
    print("🔹 开始预分词（全程仅1次，加速50倍）...")
    seg_list = []
    for text in tqdm(data_df['fusion_text'], desc="预分词中"):
        words = jieba.lcut(text.strip())
        seg_list.append(words)
    data_df['words'] = seg_list
    print("✅ 预分词完成")
    return data_df

# ===================== 词汇表 =====================
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_count = {}
        self.total_words = 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.total_words
            self.idx2word[self.total_words] = word
            self.word_count[word] = 1
            self.total_words += 1
        else:
            self.word_count[word] += 1

def build_vocabulary(data_df, max_size):
    vocab = Vocabulary()
    for words in data_df['words']:
        for w in words:
            vocab.add_word(w)
    if vocab.total_words > max_size:
        sorted_words = sorted(vocab.word_count.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:max_size - 2]
        new_vocab = Vocabulary()
        for w, _ in top_words:
            new_vocab.add_word(w)
        return new_vocab
    return vocab

def get_max_sequence_length(data_df):
    lengths = [len(ws) for ws in data_df['words']]
    return int(np.percentile(lengths, 95))

# ===================== 数据集 =====================
class DLDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.labels = df['label'].values
        self.words_list = df['words'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        words = self.words_list[idx]
        ids = [self.vocab.word2idx.get(w, 1) for w in words]
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids += [0] * (self.max_len - len(ids))
        return {
            'text': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ===================== 模型 =====================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.emb(x))
        out, _ = self.lstm(x)
        att_w = torch.softmax(self.attn(out), dim=1)
        feat = (att_w * out).sum(dim=1)
        feat = self.norm(feat)
        return self.fc(self.drop(feat))

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, 128, (k, embed_dim)) for k in [2, 3, 4]])
        self.norm = nn.LayerNorm(128 * 3)
        self.fc = nn.Linear(128 * 3, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.emb(x).unsqueeze(1)
        pools = []
        for conv in self.convs:
            o = torch.relu(conv(x)).squeeze(-1)
            pools.append(torch.max_pool1d(o, o.size(-1)).squeeze(-1))
        feat = torch.cat(pools, dim=1)
        feat = self.norm(feat)
        return self.fc(self.drop(feat))

class BiLSTMCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.emb(x))
        out, _ = self.lstm(x)
        out = out.permute(0, 2, 1)
        out = torch.relu(self.conv(out))
        feat = torch.max(out, dim=-1)[0]
        feat = self.norm(feat)
        return self.fc(self.drop(feat))

# ===================== 训练&评估 =====================
def train_one_epoch(model, loader, cri, opt, dev):
    model.train()
    loss, corr, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="训练中", leave=False):
        x, y = batch['text'].to(dev), batch['label'].to(dev)
        o = model(x)
        l = cri(o, y)
        opt.zero_grad()
        l.backward()
        opt.step()
        loss += l.item() * x.size(0)
        corr += (o.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss / total, corr / total

@torch.no_grad()
def eval_model(model, loader, cri, dev):
    model.eval()
    loss, corr, total = 0.0, 0, 0
    preds, ys = [], []
    for batch in tqdm(loader, desc="验证中", leave=False):
        x, y = batch['text'].to(dev), batch['label'].to(dev)
        o = model(x)
        loss += cri(o, y).item() * x.size(0)
        corr += (o.argmax(1) == y).sum().item()
        total += x.size(0)
        preds.extend(o.argmax(1).cpu().numpy())
        ys.extend(y.cpu().numpy())
    acc = corr / total
    macro = f1_score(ys, preds, average='macro')
    weighted = f1_score(ys, preds, average='weighted')
    return loss / total, acc, macro, weighted

# ===================== 主函数 =====================
def main():
    print(f"🚀 运行设备: {Config.DEVICE}")
    anno = load_annotation_data(Config.ANNOTATION_PATH)
    solu = load_solution_data(Config.SOLUTION_PATH)
    df = reconstruct_data_for_kv_retrieval(anno, solu)

    # 预分词（关键！！！）
    df = pre_segment_all_texts(df)

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['value'])
    n_classes = len(le.classes_)

    # 构建词表 & 序列长度
    vocab = build_vocabulary(df, Config.MAX_VOCAB_SIZE)
    max_len = get_max_sequence_length(df)
    print(f"✅ 词汇表大小: {vocab.total_words}, 最大序列长度: {max_len}")

    # 交叉验证
    kf = StratifiedKFold(Config.N_FOLDS, shuffle=True, random_state=42)
    results = {m: {'acc': [], 'macro': [], 'weighted': []} for m in Config.MODELS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(df, df['label']), 1):
        print(f"\n==================== FOLD {fold} ====================")
        tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]

        tr_loader = DataLoader(DLDataset(tr_df, vocab, max_len), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
        va_loader = DataLoader(DLDataset(va_df, vocab, max_len), batch_size=Config.BATCH_SIZE, num_workers=0)

        for m_name in Config.MODELS:
            print(f"\n🎯 训练模型: {m_name}")
            if m_name == 'BiLSTM-Attention':
                model = BiLSTMAttention(vocab.total_words, Config.EMBEDDING_DIM, Config.HIDDEN_DIM, n_classes,
                                        Config.NUM_LAYERS, Config.DROPOUT)
            elif m_name == 'TextCNN':
                model = TextCNN(vocab.total_words, Config.EMBEDDING_DIM, n_classes, Config.DROPOUT)
            else:
                model = BiLSTMCNN(vocab.total_words, Config.EMBEDDING_DIM, Config.HIDDEN_DIM, n_classes,
                                  Config.NUM_LAYERS, Config.DROPOUT)

            model = model.to(Config.DEVICE)
            cri = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

            best_acc = best_macro = best_weighted = 0.0
            for ep in range(Config.NUM_EPOCHS):
                tr_loss, tr_acc = train_one_epoch(model, tr_loader, cri, opt, Config.DEVICE)
                va_loss, va_acc, va_macro, va_weighted = eval_model(model, va_loader, cri, Config.DEVICE)

                if va_acc > best_acc:
                    best_acc = va_acc
                    best_macro = va_macro
                    best_weighted = va_weighted

                print(f"Epoch {ep+1:2d} | trLoss {tr_loss:.3f} | trAcc {tr_acc:.3f} | "
                      f"valLoss {va_loss:.3f} | valAcc {va_acc:.3f} | MacroF1 {va_macro:.3f}")

            results[m_name]['acc'].append(best_acc)
            results[m_name]['macro'].append(best_macro)
            results[m_name]['weighted'].append(best_weighted)

    # ===================== 最终输出 =====================
    print("\n" + "="*60)
    print("                最终结果（均值±标准差）")
    print("="*60)
    for m in Config.MODELS:
        acc_mean, acc_std = np.mean(results[m]['acc']), np.std(results[m]['acc'])
        macro_mean, macro_std = np.mean(results[m]['macro']), np.std(results[m]['macro'])
        w_mean, w_std = np.mean(results[m]['weighted']), np.std(results[m]['weighted'])
        print(f"\n{m}:")
        print(f"  准确率:      {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Macro-F1:    {macro_mean:.4f} ± {macro_std:.4f}")
        print(f"  Weighted-F1: {w_mean:.4f} ± {w_std:.4f}")

if __name__ == "__main__":
    main()