import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
from transformers import BertTokenizer

# 忽略警告
warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
class Config:
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"
    SOLUTION_PATH = "question_with_knowledge_solutions.json"
    MAX_VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5
    N_FOLDS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10   #与我自己使用的模型训练批次一样
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = ['BiLSTM-Attention', 'TextCNN', 'BiLSTM-CNN']
    SAVE_DIR = 'dl_baseline_visualizations_bert_solution'
    BERT_MODEL_NAME = 'bert-base-chinese'
    BERT_MAX_LENGTH = None

# 数据加载与预处理
def load_annotation_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"标注文件不存在：{path}")

    try:
        df = pd.read_csv(
            path,
            encoding='utf-8',
            sep=',',
            quotechar='"',
            escapechar='\\'
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding='gbk',
            sep=',',
            quotechar='"',
            escapechar='\\'
        )

    df.columns = [col.strip() for col in df.columns]
    required_cols = ['问题id', '知识点名称', '认知目标层次', '知识点类型']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"标注数据缺少必要字段，需包含：{required_cols}")

    df = df.drop_duplicates(subset=['问题id', '知识点名称'], keep='first')
    print(f"✅ 加载标注数据：{len(df)}条记录，认知层次类别：{df['认知目标层次'].unique().tolist()}")
    return df

def load_solution_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"解题步骤文件不存在：{path}")

    with open(path, encoding ='utf-8') as f:
        solution_json = json.load(f)

    solution_df = pd.DataFrame(solution_json['data_list'])[[
        'question_id', 'solution_steps', 'knowledge_points'
    ]].rename(columns={
        'question_id': '问题id',
        'knowledge_points': '知识点列表',
        'solution_steps': '解题步骤'
    })

    solution_df['解题步骤'] = solution_df['解题步骤'].fillna("")
    solution_df['解题步骤'] = solution_df['解题步骤'].apply(
        lambda x: "；".join(x) if isinstance(x, list) else str(x)
    )

    solution_df = solution_df.drop_duplicates(subset=['问题id'], keep='first')
    print(f"✅ 加载解题步骤数据：{len(solution_df)}个唯一问题")
    return solution_df

def reconstruct_data_for_kv_retrieval(annotation_df: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
    annotation_df['问题id'] = annotation_df['问题id'].astype(str).str.strip()
    annotation_df['知识点名称'] = annotation_df['知识点名称'].astype(str).str.strip()
    solution_df['问题id'] = solution_df['问题id'].astype(str).str.strip()

    merged_df = pd.merge(
        annotation_df,
        solution_df[['问题id', '解题步骤']],
        on=['问题id'],
        how='inner'
    )

    merged_df = merged_df[
        (merged_df['解题步骤'].notna()) &
        (merged_df['解题步骤'].str.strip() != '') &
        (merged_df['认知目标层次'].notna()) &
        (merged_df['认知目标层次'].str.strip() != '')
        ]

    processed_data = []
    for question_id, group in merged_df.groupby('问题id'):
        solution_text = group['解题步骤'].iloc[0]
        for _, row in group.iterrows():
            knowledge_point = f"{row['知识点名称']}"
            cognitive_level = row['认知目标层次']
            processed_data.append({
                'question_id': question_id,
                'query': solution_text,
                'key': knowledge_point,
                'value': cognitive_level,
                'fusion_text': f"解题步骤:{solution_text}；知识点:{knowledge_point}"
            })

    processed_df = pd.DataFrame(processed_data)
    print(f"✅ 重构后数据量：{len(processed_df)}条记录")
    return processed_df

# 文本预处理与词汇表构建（BERT Tokenizer）
class Vocabulary:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.word2idx = self.tokenizer.vocab
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.total_words = len(self.word2idx)

    def get_word_index(self, word):
        return self.word2idx.get(word, self.tokenizer.unk_token_id)

    # ====================== 这里是关键 ======================
    def get_sentence_indices(self, sentence, max_length):
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='np'
        )
        return encoding['input_ids'].squeeze()

def build_vocabulary(data_df, max_vocab_size):
    vocab = Vocabulary()
    print(f"✅ BERT词汇表构建完成，包含 {vocab.total_words} 个词")
    return vocab

def get_max_sequence_length(data_df, percentile=95):
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    lengths = [
        len(tokenizer.encode(text, add_special_tokens=True))
        for text in data_df['fusion_text']
    ]
    max_len = int(np.percentile(lengths, percentile))
    max_len = min(max_len, 512)
    Config.BERT_MAX_LENGTH = max_len
    print(f"✅ 序列长度统计：平均={np.mean(lengths):.1f}, 中位数={np.median(lengths):.1f}, 95分位={max_len}")
    return max_len

# 数据集类
class DLDataset(Dataset):
    def __init__(self, data_df, vocab, max_length, label_encoder=None):
        self.data_df = data_df
        self.vocab = vocab
        self.max_length = max_length

        if label_encoder is not None:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(data_df['value'])
        else:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(data_df['value'])
        self.classes = self.label_encoder.classes_

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row['fusion_text']
        text_indices = self.vocab.get_sentence_indices(text, self.max_length)
        label = self.labels[idx]

        return {
            'text': torch.tensor(text_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 模型定义
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)

        query_cls = lstm_out[:, 0, :]
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_cls = torch.sum(attention_weights * lstm_out, dim=1)

        fused_features = torch.cat([query_cls, attended_cls], dim=1)
        fused_features = self.layer_norm(fused_features)

        output = self.fc(self.dropout(fused_features))
        return output

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, num_filters=128, filter_sizes=[2, 3, 4], dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_filters * len(filter_sizes) * 2)
        self.fc = nn.Linear(num_filters * len(filter_sizes) * 2, num_classes)

    def forward(self, text):
        embedded = self.embedding(text).unsqueeze(1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded)).squeeze(3)
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pool_out)
        query_cls = torch.cat(conv_outputs, dim=1)

        attended_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded)).squeeze(3)
            attn_out = torch.mean(conv_out, dim=2)
            attended_outputs.append(attn_out)
        attended_cls = torch.cat(attended_outputs, dim=1)

        fused_features = torch.cat([query_cls, attended_cls], dim=1)
        fused_features = self.layer_norm(fused_features)

        output = self.fc(self.dropout(fused_features))
        return output

class BiLSTMCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers, dropout):
        super(BiLSTMCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.conv = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        self.fc = nn.Linear(hidden_dim * 3, num_classes)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, _ = self.lstm(embedded)

        query_cls = lstm_out[:, 0, :]

        lstm_out_permuted = lstm_out.permute(0, 2, 1)
        cnn_out = torch.relu(self.conv(lstm_out_permuted))
        attended_cls = torch.max(cnn_out, dim=2)[0]

        fused_features = torch.cat([query_cls, attended_cls], dim=1)
        fused_features = self.layer_norm(fused_features)

        output = self.fc(self.dropout(fused_features))
        return output

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name, fold):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_preds = []
    best_labels = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            text = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * text.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                text = batch['text'].to(device)
                labels = batch['label'].to(device)

                outputs = model(text)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * text.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            torch.save(model.state_dict(), f'{Config.SAVE_DIR}/{model_name}_fold{fold}_best.pth')

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, fold)

    return best_val_acc, best_labels, best_preds

# 可视化函数
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, fold):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{model_name} Fold {fold} 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.title(f'{model_name} Fold {fold} 准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{Config.SAVE_DIR}/{model_name}_fold{fold}_training_curves.png')
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, classes, model_name, fold):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title(f'{model_name} Fold {fold} 混淆矩阵（归一化）', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{Config.SAVE_DIR}/{model_name}_fold{fold}_confusion_matrix.png')
    plt.close()

# 主函数
def main():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    annotation_df = load_annotation_data(Config.ANNOTATION_PATH)
    solution_df = load_solution_data(Config.SOLUTION_PATH)
    data_df = reconstruct_data_for_kv_retrieval(annotation_df, solution_df)

    kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    
    results = {
        model_name: {
            "acc": [], "macro_f1": [], "weighted_f1": []
        } for model_name in Config.MODELS
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df, data_df['value']), 1):
        print(f"\n{'=' * 50}")
        print(f"开始第 {fold} 折交叉验证")
        print(f"{'=' * 50}")

        train_df = data_df.iloc[train_idx]
        val_df = data_df.iloc[val_idx]

        vocab = build_vocabulary(train_df, Config.MAX_VOCAB_SIZE)
        max_length = get_max_sequence_length(train_df)

        train_dataset = DLDataset(train_df, vocab, max_length)
        val_dataset = DLDataset(val_df, vocab, max_length, label_encoder=train_dataset.label_encoder)
        classes = train_dataset.classes

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)

        for model_name in Config.MODELS:
            print(f"\n--- 训练 {model_name} 模型 ---")

            if model_name == 'BiLSTM-Attention':
                model = BiLSTMAttention(
                    vocab_size=vocab.total_words,
                    embedding_dim=Config.EMBEDDING_DIM,
                    hidden_dim=Config.HIDDEN_DIM,
                    num_classes=len(classes),
                    num_layers=Config.NUM_LAYERS,
                    dropout=Config.DROPOUT
                )
            elif model_name == 'TextCNN':
                model = TextCNN(
                    vocab_size=vocab.total_words,
                    embedding_dim=Config.EMBEDDING_DIM,
                    num_classes=len(classes),
                    dropout=Config.DROPOUT
                )
            elif model_name == 'BiLSTM-CNN':
                model = BiLSTMCNN(
                    vocab_size=vocab.total_words,
                    embedding_dim=Config.EMBEDDING_DIM,
                    hidden_dim=Config.HIDDEN_DIM,
                    num_classes=len(classes),
                    num_layers=Config.NUM_LAYERS,
                    dropout=Config.DROPOUT
                )

            model = model.to(Config.DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

            best_acc, true_labels, pred_labels = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=Config.NUM_EPOCHS,
                device=Config.DEVICE,
                model_name=model_name,
                fold=fold
            )

            acc = accuracy_score(true_labels, pred_labels)
            macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
            weighted_f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

            results[model_name]["acc"].append(acc)
            results[model_name]["macro_f1"].append(macro_f1)
            results[model_name]["weighted_f1"].append(weighted_f1)

            plot_confusion_matrix(true_labels, pred_labels, classes, model_name, fold)

            report = classification_report(true_labels, pred_labels, target_names=classes, digits=4)
            with open(f'{Config.SAVE_DIR}/{model_name}_fold{fold}_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"{model_name} 第 {fold} 折最佳指标:")
            print(f"  ACC = {acc:.4f} | Macro-F1 = {macro_f1:.4f} | Weighted-F1 = {weighted_f1:.4f}")

    print(f"\n{'=' * 65}")
    print("  所有模型 5 折交叉验证最终结果（均值 ± 标准差）")
    print(f"{'=' * 65}")

    final_results = []
    for model_name in Config.MODELS:
        acc_mean = np.mean(results[model_name]["acc"])
        acc_std = np.std(results[model_name]["acc"])
        
        macro_mean = np.mean(results[model_name]["macro_f1"])
        macro_std = np.std(results[model_name]["macro_f1"])
        
        weighted_mean = np.mean(results[model_name]["weighted_f1"])
        weighted_std = np.std(results[model_name]["weighted_f1"])

        final_results.append({
            'model': model_name,
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'macro_mean': macro_mean, 'macro_std': macro_std,
            'weighted_mean': weighted_mean, 'weighted_std': weighted_std,
        })

        print(f"\n【{model_name}】")
        print(f"  ACC          = {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Macro-F1     = {macro_mean:.4f} ± {macro_std:.4f}")
        print(f"  Weighted-F1  = {weighted_mean:.4f} ± {weighted_std:.4f}")

    with open(f'{Config.SAVE_DIR}/final_results.txt', 'w', encoding='utf-8') as f:
        f.write("深度学习基线模型交叉验证结果（BERT分词+解题步骤输入）\n")
        f.write("=" * 60 + "\n")
        for res in final_results:
            f.write(f"\n【{res['model']}】\n")
            f.write(f"ACC         = {res['acc_mean']:.4f} ± {res['acc_std']:.4f}\n")
            f.write(f"Macro-F1    = {res['macro_mean']:.4f} ± {res['macro_std']:.4f}\n")
            f.write(f"Weighted-F1 = {res['weighted_mean']:.4f} ± {res['weighted_std']:.4f}\n")

    print(f"\n✅ 所有结果已保存至：{Config.SAVE_DIR}")

if __name__ == "__main__":
    main()
