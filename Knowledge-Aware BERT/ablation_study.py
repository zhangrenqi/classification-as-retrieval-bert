#添加可视化要求：
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
from collections import defaultdict
import pickle
# 可视化相关库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# # 忽略警告
# warnings.filterwarnings('ignore')
# # 设置中文字体（解决中文乱码）
# plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
# plt.rcParams['axes.unicode_minus'] = False

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.environ['MATPLOTLIB_VERBOSITY'] = 'quiet'

# import matplotlib
# matplotlib.rcParams['backend'] = 'Agg'  # 无界面模式
# matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False


import matplotlib
matplotlib.rcParams['backend'] = 'Agg'  # 保持无界面模式
# 设置为通用的英文字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


# 关闭所有日志
import logging
logging.getLogger('matplotlib').setLevel(logging.FATAL)
logging.getLogger('PIL').setLevel(logging.FATAL)

import warnings
warnings.filterwarnings("ignore")

# 创建可视化结果保存文件夹
os.makedirs('visualizations6', exist_ok=True)

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------- 1. 配置参数 --------------------------
class Config:
    # 数据路径
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"  # 标注数据集
    SOLUTION_PATH = "question_with_knowledge_solutions.json"  # 解题步骤数据集
    # BERT配置
    BERT_MODEL = "bert-base-chinese"  # 中文BERT模型
    MAX_QUERY_LEN = 512  # 解题过程最大长度
    MAX_KEY_LEN = 128     # 知识点最大长度
    # 模型训练配置
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5   #1e-5/2e-5试验过都不好
    HIDDEN_DIM = 256  # 分类头隐藏层维度
    # 交叉验证配置
    N_FOLDS = 5  # 五折交叉验证
    NUM_EPOCHS = 10
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 可视化配置（仅保留2D TSNE参数）
    TSNE_PERPLEXITY = 30  # TSNE降维参数
    TSNE_max_iter = 1000   # TSNE迭代次数
    PLOT_DPI = 300         # 图片分辨率
    
    # 消融实验模式配置 (1~6)
    # 1: 仅原题文本 (Single)
    # 2: 仅解题过程 (Single)
    # 3: 仅知识点 (Single)
    # 4: 解题过程 + 知识点 (Dual, Baseline)
    # 5: 原题文本 + 知识点 (Dual)
    # 6: 原题文本 + 解题过程 + 知识点 (Dual, 拼接题与解为Query)
    ABLATION_MODE = 6

    # 认知层次中英文映射
    LEVEL_EN_MAP = {
        "记忆": "Remember",
        "理解": "Understand",
        "应用": "Apply",
        "分析": "Analyze",
        "评价与创造": "Evaluate & Create",
    }





# -------------------------- 2. 数据加载与预处理 --------------------------
def load_annotation_data(path: str) -> pd.DataFrame:
    """加载标注数据（包含认知目标层次）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"标注文件不存在：{path}")

    # 读取CSV格式的标注数据
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

    # 处理列名空格
    df.columns = [col.strip() for col in df.columns]

    # 保留必要字段
    required_cols = ['问题id', '知识点名称', '认知目标层次', '知识点类型']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"标注数据缺少必要字段，需包含：{required_cols}，实际包含：{df.columns.tolist()}")

    # 去重（同一问题+知识点的重复标注）
    df = df.drop_duplicates(subset=['问题id', '知识点名称'], keep='first')
    print(f"✅ 加载标注数据：{len(df)}条记录，认知层次类别：{df['认知目标层次'].unique().tolist()}")
    return df

# def load_solution_data(path: str) -> pd.DataFrame:
#     """加载解题步骤数据（按问题id唯一存储）"""
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"解题步骤文件不存在：{path}")

#     with open(path, 'r', encoding='utf-8') as f:
#         solution_json = json.load(f)

#     # 转换为DataFrame（每个问题id只保留一条记录，含完整解题步骤）
#     solution_df = pd.DataFrame(solution_json['data_list'])[[
#         'question_id', 'solution_steps', 'knowledge_points'
#     ]].rename(columns={
#         'question_id': '问题id',
#         'knowledge_points': '知识点列表'  # 保留原始知识点列表用于校验
#     })

#     # 确保问题id唯一（一个问题对应一个解题步骤）
#     solution_df = solution_df.drop_duplicates(subset=['问题id'], keep='first')
#     print(f"✅ 加载解题步骤数据：{len(solution_df)}个唯一问题（每个问题一个解题步骤）")
#     return solution_df

# def reconstruct_data_for_kv_retrieval(annotation_df: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     重构数据格式，适应键值检索需求
#     Query: 解题过程
#     Key: 知识点
#     Value: 认知目标层次
#     """
#     # 统一字符串格式
#     annotation_df['问题id'] = annotation_df['问题id'].astype(str).str.strip()
#     annotation_df['知识点名称'] = annotation_df['知识点名称'].astype(str).str.strip()
#     solution_df['问题id'] = solution_df['问题id'].astype(str).str.strip()

#     # 按问题id合并
#     merged_df = pd.merge(
#         annotation_df,
#         solution_df[['问题id', 'solution_steps']],
#         on=['问题id'],
#         how='inner'
#     ).rename(columns={'solution_steps': '解题步骤'})

#     # 过滤无效数据
#     merged_df = merged_df[
#         (merged_df['解题步骤'].notna()) &
#         (merged_df['解题步骤'].str.strip() != '') &
#         (merged_df['认知目标层次'].notna()) &
#         (merged_df['认知目标层次'].str.strip() != '')
#     ]

#     processed_data = []
    
#     # 按问题分组，重构数据格式
#     for question_id, group in merged_df.groupby('问题id'):
#         # 解题过程作为query（相同的查询内容）
#         solution_text = group['解题步骤'].iloc[0]  # 每个问题只有一个解题过程
        
#         # 该问题的每个知识点作为独立的key-value对
#         for _, row in group.iterrows():
#             knowledge_point = f"{row['知识点名称']}"
#             cognitive_level = row['认知目标层次']
            
#             processed_data.append({
#                 'question_id': question_id,
#                 'query': solution_text,           # Query: 解题过程
#                 'key': knowledge_point,           # Key: 知识点
#                 'value': cognitive_level,         # Value: 解题过程+知识点
#                 'fusion_text': f"解题过程:{solution_text}；知识点:{knowledge_point}"
#             })
    
#     processed_df = pd.DataFrame(processed_data)
    
#     # 统计信息
#     print(f"✅ 重构后数据量：{len(processed_df)}条记录")
#     print(f"📊 涉及问题数量：{processed_df['question_id'].nunique()}个")
#     print(f"📊 平均每个问题包含知识点：{len(processed_df) / processed_df['question_id'].nunique():.2f}个")
#     print("认知层次分布：")
#     print(processed_df['value'].value_counts().to_string())
    
#     return processed_df
def load_solution_data(path: str) -> pd.DataFrame:
    """加载解题步骤数据（更新：增加加载 original_question）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"解题步骤文件不存在：{path}")

    with open(path, 'r', encoding='utf-8') as f:
        solution_json = json.load(f)

    # 提取必要的字段，包括原题文本
    solution_df = pd.DataFrame(solution_json['data_list'])[[
        'question_id', 'original_question', 'solution_steps', 'knowledge_points'
    ]].rename(columns={
        'question_id': '问题id',
        'original_question': '原题文本',
        'knowledge_points': '知识点列表'
    })

    solution_df = solution_df.drop_duplicates(subset=['问题id'], keep='first')
    print(f"✅ 加载解题步骤数据：{len(solution_df)}个唯一问题")
    return solution_df

def reconstruct_data_for_kv_retrieval(annotation_df: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
    """根据消融实验模式重构数据格式"""
    # 统一字符串格式
    annotation_df['问题id'] = annotation_df['问题id'].astype(str).str.strip()
    annotation_df['知识点名称'] = annotation_df['知识点名称'].astype(str).str.strip()
    solution_df['问题id'] = solution_df['问题id'].astype(str).str.strip()

    # 按问题id合并，现在包含了 '原题文本'
    merged_df = pd.merge(
        annotation_df,
        solution_df[['问题id', '原题文本', 'solution_steps']],
        on=['问题id'],
        how='inner'
    ).rename(columns={'solution_steps': '解题步骤'})

    processed_data = []
    mode = Config.ABLATION_MODE
    
    for question_id, group in merged_df.groupby('问题id'):
        q_text = group['原题文本'].iloc[0]
        s_text = group['解题步骤'].iloc[0]
        
        for _, row in group.iterrows():
            k_text = f"{row['知识点名称']}"
            cognitive_level = row['认知目标层次']
            # 转换为英文，如果没找到则保留原样
            cognitive_level_en = Config.LEVEL_EN_MAP.get(cognitive_level, cognitive_level)
            
            # 根据模式动态分配 Query 和 Key
            if mode == 1:
                query, key = q_text, ""
            elif mode == 2:
                query, key = s_text, ""
            elif mode == 3:
                query, key = k_text, ""
            elif mode == 4:
                query, key = s_text, k_text
            elif mode == 5:
                query, key = q_text, k_text
            elif mode == 6:
                query, key = f"题目：{q_text}；解题过程：{s_text}", k_text
            
            processed_data.append({
                'question_id': question_id,
                'query': query,           
                'key': key,           
                'value': cognitive_level_en, 
            })
    
    processed_df = pd.DataFrame(processed_data)
    print(f"✅ 消融模式 [{mode}] 数据重构完成，总数据量：{len(processed_df)}条")
    return processed_df
# -------------------------- 3. 数据集与数据加载器 --------------------------
class KVCognitiveDataset(Dataset):
    """基于键值检索的认知层次分类数据集"""
    def __init__(self, queries, keys, labels, tokenizer, max_query_len, max_key_len):
        self.queries = queries  # 解题过程列表
        self.keys = keys        # 知识点列表  
        self.labels = labels    # 认知层次标签列表
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_key_len = max_key_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_text = str(self.queries[idx])
        # 保护机制：如果 key 为空，用 "[PAD]" 占位，防止 Tokenizer 报错
        key_text = str(self.keys[idx]) if str(self.keys[idx]).strip() != "" else "[PAD]" 
        label = self.labels[idx]

        # 分别编码query（解题过程）和key（知识点）
        query_encoding = self.tokenizer(
            query_text,
            add_special_tokens=True,
            max_length=self.max_query_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        key_encoding = self.tokenizer(
            key_text,
            add_special_tokens=True,
            max_length=self.max_key_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoding['input_ids'].flatten(),
            'query_attention_mask': query_encoding['attention_mask'].flatten(),
            'key_input_ids': key_encoding['input_ids'].flatten(),
            'key_attention_mask': key_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'query_text': query_text,  # 用于注意力可视化
            'key_text': key_text       # 用于注意力可视化
        }

def create_kv_data_loader(queries, keys, labels, tokenizer, max_query_len, max_key_len, batch_size, shuffle=True):
    """创建键值检索数据加载器"""
    ds = KVCognitiveDataset(
        queries=queries,
        keys=keys,
        labels=labels,
        tokenizer=tokenizer,
        max_query_len=max_query_len,
        max_key_len=max_key_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4 if os.name != 'nt' else 0,  
        pin_memory=True
    )

# -------------------------- 4. 模型定义 --------------------------
class KnowledgeAwareCognitiveClassifier(nn.Module):
    """支持消融实验的认知层次分类模型"""
    def __init__(self, bert_model_name, num_classes, hidden_dim=256):
        super().__init__()
        self.mode = Config.ABLATION_MODE
        self.is_single_input = self.mode in [1, 2, 3] # 判断是否为单输入模式
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        if not self.is_single_input:
            # 双输入模式：保留注意力机制
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=self.bert.config.hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            classifier_input_dim = self.bert.config.hidden_size * 2
            self.layer_norm = nn.LayerNorm(classifier_input_dim)
        else:
            # 单输入模式：只用一个 BERT 输出
            classifier_input_dim = self.bert.config.hidden_size
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, query_input_ids, query_attention_mask, key_input_ids=None, key_attention_mask=None, return_features=False):
        
        # 无论什么模式，Query 都会被送入 BERT
        query_outputs = self.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embeddings = query_outputs.last_hidden_state  
        query_cls = query_embeddings[:, 0, :] 
        
        if self.is_single_input:
            # 单输入模式下，直接利用 Query 的 [CLS] 向量进行分类
            logits = self.classifier(query_cls)
            if return_features:
                # 注意：单输入模式没有 attention_weights 返回
                return logits, None, query_cls 
            return logits, None

        # --- 以下为双输入模式 (4, 5, 6) 的逻辑 ---
        key_outputs = self.bert(
            input_ids=key_input_ids,
            attention_mask=key_attention_mask
        )
        key_embeddings = key_outputs.last_hidden_state 
        
        attended_output, attention_weights = self.attention_layer(
            query=query_embeddings,
            key=key_embeddings,
            value=key_embeddings,
            key_padding_mask=~key_attention_mask.bool() if key_attention_mask is not None else None
        )
        
        attended_cls = attended_output[:, 0, :] 
        fused_features = torch.cat([query_cls, attended_cls], dim=1)
        fused_features = self.layer_norm(fused_features)
        
        logits = self.classifier(fused_features)
        
        if return_features:
            return logits, attention_weights, fused_features 
        return logits, attention_weights

# -------------------------- 5. 保存和加载辅助函数 --------------------------
def save_checkpoint(model, optimizer, label_encoder, epoch, val_acc, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'label_encoder_classes': label_encoder.classes_,
        'epoch': epoch,
        'val_acc': val_acc
    }
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint保存成功: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    except:
        print("⚠️ 使用weights_only=False加载checkpoint（确保文件来源可信）")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = checkpoint['label_encoder_classes']
    
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint加载成功: epoch={checkpoint['epoch']}, val_acc={checkpoint['val_acc']:.4f}")
    return model, label_encoder, checkpoint['epoch'], checkpoint['val_acc']

def save_label_encoder(label_encoder, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'classes_': label_encoder.classes_}, f)
    print(f"✅ LabelEncoder保存成功: {filename}")

def load_label_encoder(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = data['classes_']
    return label_encoder

# -------------------------- 6. 可视化工具类（仅保留2D TSNE） --------------------------
class Visualizer:
    """实现5项可视化功能（删除3D TSNE）"""
    def __init__(self, label_encoder, save_dir='visualizations6', dpi=Config.PLOT_DPI):
        self.label_encoder = label_encoder
        self.save_dir = save_dir
        self.dpi = dpi
        self.classes = label_encoder.classes_
        self.n_classes = len(self.classes)
        # 配色方案（支持多类别）
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        if self.n_classes > len(self.colors):
            self.colors = plt.cm.tab10(np.linspace(0, 1, self.n_classes))  # 自动扩展颜色

    def plot_loss_curve(self, fold_train_losses, fold_val_losses, fold_idx):
        """1. 可视化损失函数曲线"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(fold_train_losses) + 1)
        
        plt.plot(epochs, fold_train_losses, 'o-', color=self.colors[0], label='Training loss', linewidth=2)
        plt.plot(epochs, fold_val_losses, 's-', color=self.colors[1], label='Validation loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title(f'Fold {fold_idx} Training and Validation Loss Curves', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fold_{fold_idx}_loss_curve.png", dpi=self.dpi)
        plt.close()

    def plot_accuracy_curve(self, fold_train_accs, fold_val_accs, fold_idx):
        """2. 可视化准确率曲线"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(fold_train_accs) + 1)
        
        plt.plot(epochs, fold_train_accs, 'o-', color=self.colors[2], label='Training accuracy', linewidth=2)
        plt.plot(epochs, fold_val_accs, 's-', color=self.colors[3], label='Validation accuracy', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Fold {fold_idx} Training and Validation Accuracy Curves', fontsize=14)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fold_{fold_idx}_accuracy_curve.png", dpi=self.dpi)
        plt.close()

    def plot_class_accuracy(self, true_labels, pred_labels, fold_idx):
        """3. 生成类别准确率"""
        # 计算每个类别的准确率
        class_acc = {}
        for cls_idx, cls_name in enumerate(self.classes):
            mask = (np.array(true_labels) == cls_idx)
            if np.sum(mask) == 0:
                class_acc[cls_name] = 0.0
                continue
            class_acc[cls_name] = np.mean(np.array(pred_labels)[mask] == cls_idx)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        cls_names = list(class_acc.keys())
        acc_values = [class_acc[name] for name in cls_names]
        
        sns.barplot(x=cls_names, y=acc_values, palette=self.colors[:len(cls_names)])
        plt.xlabel('Cognitive hierarchy category', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Fold {fold_idx} Class-wise Accuracy', fontsize=14)
        plt.ylim(0, 1.05)
        # 在柱状图上标注准确率
        for i, v in enumerate(acc_values):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fold_{fold_idx}_class_accuracy.png", dpi=self.dpi)
        plt.close()

    def plot_confusion_matrix(self, true_labels, pred_labels, fold_idx):
        """4. 生成混淆矩阵"""
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels, labels=range(self.n_classes))
        # 归一化（按行）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 可视化
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.xlabel('Prediction category', fontsize=12)
        plt.ylabel('True category', fontsize=12)
        plt.title(f'Fold {fold_idx} Confusion Matrix (Normalized)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/fold_{fold_idx}_confusion_matrix.png", dpi=self.dpi)
        plt.close()

    def plot_attention_weights(self, model, data_loader, device, fold_idx, num_samples=3):
        if Config.ABLATION_MODE in [1, 2, 3]:
            print(f"⚠️ 模式 {Config.ABLATION_MODE} 无交叉注意力机制，跳过注意力热力图绘制。")
            return
        """5. 注意力权重可视化（展示query与key的交互）"""
        model.eval()
        samples_plotted = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if samples_plotted >= num_samples:
                    break
                    
                # 取第一个样本
                query_input_ids = batch["query_input_ids"][:1].to(device)
                query_attention_mask = batch["query_attention_mask"][:1].to(device)
                key_input_ids = batch["key_input_ids"][:1].to(device)
                key_attention_mask = batch["key_attention_mask"][:1].to(device)
                query_text = batch["query_text"][0]
                key_text = batch["key_text"][0]
                
                # 获取注意力权重
                _, attention_weights, _ = model(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    key_input_ids=key_input_ids,
                    key_attention_mask=key_attention_mask,
                    return_features=True
                )
                
                # 注意力权重形状：[num_heads, batch_size, query_len, key_len]
                # 取第一个头的权重（平均所有头的权重）
                attention_weights = attention_weights.mean(dim=0).squeeze(0).cpu().numpy()  # [query_len, key_len]
                
                # 解码token（去除padding）
                query_tokens = data_loader.dataset.tokenizer.convert_ids_to_tokens(query_input_ids[0].cpu().numpy())
                key_tokens = data_loader.dataset.tokenizer.convert_ids_to_tokens(key_input_ids[0].cpu().numpy())
                
                # 过滤padding（[PAD]）
                query_tokens = [t for t in query_tokens if t != '[PAD]']
                key_tokens = [t for t in key_tokens if t != '[PAD]']
                query_len = len(query_tokens)
                key_len = len(key_tokens)
                attention_weights = attention_weights[:query_len, :key_len]  # 截取有效长度
                
                # 可视化注意力热力图
                plt.figure(figsize=(12, 8))
                sns.heatmap(attention_weights, annot=False, cmap='viridis', 
                           xticklabels=key_tokens, yticklabels=query_tokens)
                plt.xlabel('Knowledge point: Token', fontsize=12)
                plt.ylabel('Solution steps: Token', fontsize=12)
                plt.title(f'Fold {fold_idx} Attention weight heat map', fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.save_dir}/fold_{fold_idx}_attention_sample_{samples_plotted+1}.png", dpi=self.dpi)
                plt.close()
                
                samples_plotted += 1

    def plot_tsne_features(self, features, labels, fold_idx):
        """6. TSNE降维可视化BERT特征向量（仅保留2D）"""
        # 检查输入数据有效性
        if features is None or len(features) == 0:
            print(f"⚠️ TSNE可视化失败：特征为空（长度{len(features) if features is not None else 0}）")
            return
        if len(labels) != len(features):
            print(f"⚠️ TSNE可视化失败：特征数量({len(features)})与标签数量({len(labels)})不匹配")
            return
        
        # 确保特征数量适中（过多会变慢）
        if len(features) > 1000:
            print(f"⚠️ TSNE可视化：特征数量过多（{len(features)}），随机采样1000个样本")
            indices = np.random.choice(len(features), 1000, replace=False)
            features = features[indices]
            labels = np.array(labels)[indices]
        else:
            labels = np.array(labels)  # 确保标签是numpy数组
        
        # 检查TSNE输入格式（必须是2D数组）
        if len(features.shape) != 2:
            print(f"⚠️ TSNE可视化失败：特征不是2D数组（形状{features.shape}），自动展平")
            features = features.reshape(len(features), -1)  # 展平为[样本数, 特征维度]
        
        try:
            # TSNE降维（固定随机种子确保可复现）
            tsne = TSNE(
                n_components=2,  # 仅保留2D
                perplexity=Config.TSNE_PERPLEXITY,
                max_iter=Config.TSNE_max_iter,
                random_state=SEED,
                init='pca'  # 用PCA初始化，加速收敛并减少异常点
            )
            features_tsne = tsne.fit_transform(features)
            print(f"✅ TSNE降维完成：{features.shape} → {features_tsne.shape}")
        except Exception as e:
            print(f"❌ TSNE降维失败：{str(e)}")
            return
        
        # 2D绘图
        plt.figure(figsize=(10, 8))
        for cls_idx in range(self.n_classes):
            mask = (labels == cls_idx)
            if np.sum(mask) == 0:
                continue  # 跳过无样本的类别
            plt.scatter(
                features_tsne[mask, 0], 
                features_tsne[mask, 1],
                c=[self.colors[cls_idx % len(self.colors)]],  # 避免颜色索引越界
                label=self.classes[cls_idx],
                alpha=0.7,
                s=60,  # 点大小
                edgecolors='white',  # 白色边框，区分重叠点
                linewidth=0.5
            )
        
        plt.xlabel('TSNE dimension 1', fontsize=12)
        plt.ylabel('TSNE dimension 2', fontsize=12)
        plt.title(f'Fold{fold_idx} BERT feature TSNE dimensionality Reduction Visualization (2D)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{self.save_dir}/fold_{fold_idx}_tsne_2d.png", 
            dpi=self.dpi,
            bbox_inches='tight'  # 保存时包含图例
        )
        plt.close()
        print(f"✅ TSNE图保存成功：{self.save_dir}/fold_{fold_idx}_tsne_2d.png")

# -------------------------- 7. 训练与评估函数 --------------------------
def train_kv_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="训练中"):
        # 移动到设备
        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        key_input_ids = batch["key_input_ids"].to(device)
        key_attention_mask = batch["key_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        outputs, _ = model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            key_input_ids=key_input_ids,
            key_attention_mask=key_attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        # 统计正确预测数和损失
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        # 反向传播和优化
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 计算平均损失和准确率
    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct_predictions.double() / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc

def eval_kv_model(model, data_loader, loss_fn, device, label_encoder, return_features=False):
    """评估模型（支持返回特征用于TSNE）"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_features = []  # 存储BERT特征用于TSNE

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            query_input_ids = batch["query_input_ids"].to(device)
            query_attention_mask = batch["query_attention_mask"].to(device)
            key_input_ids = batch["key_input_ids"].to(device)
            key_attention_mask = batch["key_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播（根据需要返回特征）
            if return_features:
                outputs, _, features = model(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    key_input_ids=key_input_ids,
                    key_attention_mask=key_attention_mask,
                    return_features=True
                )
                all_features.append(features.cpu().numpy())
            else:
                outputs, _ = model(
                    query_input_ids=query_input_ids,
                    query_attention_mask=query_attention_mask,
                    key_input_ids=key_input_ids,
                    key_attention_mask=key_attention_mask
                )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            correct_predictions += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct_predictions.double() / len(data_loader.dataset)
    
    # 处理特征（拼接所有批次）
    if return_features and all_features:
        all_features = np.concatenate(all_features, axis=0)
        # 检查特征形状和数值有效性
        print(f"✅ 生成特征：形状{all_features.shape}，数值范围[{all_features.min():.2f}, {all_features.max():.2f}]")
        # 过滤异常特征（如全零向量）
        valid_mask = ~np.all(all_features == 0, axis=1)
        all_features = all_features[valid_mask]
        # 同步过滤标签（确保特征和标签数量匹配）
        all_labels = np.array(all_labels)[valid_mask].tolist()
        all_preds = np.array(all_preds)[valid_mask].tolist()
        print(f"✅ 过滤后特征：形状{all_features.shape}，标签数量{len(all_labels)}")
    else:
        all_features = None
        print("⚠️ 未生成有效特征（all_features为空）")
    
    # 获取实际出现的类别
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    present_labels = np.unique(all_labels_np)
    
    # 生成分类报告
    if len(present_labels) > 0:
        present_target_names = [label_encoder.classes_[i] for i in present_labels]
        classification_rep = classification_report(
            all_labels_np, all_preds_np,
            target_names=present_target_names,
            labels=present_labels,
            zero_division=0,
            output_dict=False
        )
    else:
        classification_rep = "No samples in validation set"
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc.item(),
        'predictions': all_preds,
        'labels': all_labels,
        'features': all_features,  # BERT特征向量
        'present_labels': present_labels,
        'classification_report': classification_rep
    }

def safe_classification_report(true_labels, pred_labels, label_encoder, zero_division=0):
    """安全的分类报告生成函数"""
    true_labels_np = np.array(true_labels)
    pred_labels_np = np.array(pred_labels)
    
    present_labels = np.unique(np.concatenate([true_labels_np, pred_labels_np]))
    
    if len(present_labels) == 0:
        return "No samples available for classification report"
    
    valid_labels = present_labels[present_labels < len(label_encoder.classes_)]
    
    if len(valid_labels) == 0:
        return "No valid labels available for classification report"
    
    present_target_names = [label_encoder.classes_[i] for i in valid_labels]
    
    try:
        return classification_report(
            true_labels_np, pred_labels_np,
            target_names=present_target_names,
            labels=valid_labels,
            zero_division=zero_division
        )
    except Exception as e:
        return f"Error generating classification report: {str(e)}"

def calculate_fold_metrics_summary(fold_results):
    """计算跨折性能的均值±标准差，按指定格式输出"""
    # 提取所有折的基础指标
    fold_accs = [r['val_accuracy'] for r in fold_results]
    fold_losses = [r['val_loss_final'] for r in fold_results]
    fold_train_accs = [r['train_acc'][-1] for r in fold_results]
    fold_train_losses = [r['train_loss'][-1] for r in fold_results]
    
    # 计算F1分数（Macro-F1和Weighted-F1）
    fold_macro_f1 = []
    fold_weighted_f1 = []
    for r in fold_results:
        preds = r['val_predictions']
        labels = r['val_labels']
        # 计算Macro-F1
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        # 计算Weighted-F1
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        fold_macro_f1.append(macro_f1)
        fold_weighted_f1.append(weighted_f1)
    
    # 计算均值和标准差
    acc_mean = np.mean(fold_accs)
    acc_std = np.std(fold_accs)
    macro_f1_mean = np.mean(fold_macro_f1)
    macro_f1_std = np.std(fold_macro_f1)
    weighted_f1_mean = np.mean(fold_weighted_f1)
    weighted_f1_std = np.std(fold_weighted_f1)

    # 严格按照你要求的格式输出
    print("="*60)
    print(f"准确率:    {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Macro-F1:  {macro_f1_mean:.4f} ± {macro_f1_std:.4f}")
    print(f"Weighted-F1: {weighted_f1_mean:.4f} ± {weighted_f1_std:.4f}")
    print("="*60)
    
    metrics_summary = {
        'val_accuracy': {'mean': acc_mean, 'std': acc_std, 'all': fold_accs},
        'macro_f1': {'mean': macro_f1_mean, 'std': macro_f1_std, 'all': fold_macro_f1},
        'weighted_f1': {'mean': weighted_f1_mean, 'std': weighted_f1_std, 'all': fold_weighted_f1},
    }
    return metrics_summary

def get_class_level_performance(true_labels, pred_labels, label_encoder):
    """
    计算每个认知层次的详细性能指标
    返回：DataFrame，包含列：认知层次、精确率、召回率、F1值、样本数、样本占比
    """
    true_labels_np = np.array(true_labels)
    pred_labels_np = np.array(pred_labels)
    
    # 计算每个类别的精确率、召回率、F1值
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels_np, 
        pred_labels_np, 
        labels=range(len(label_encoder.classes_)),
        zero_division=0
    )
    
    # 计算样本占比
    total_samples = np.sum(support)
    sample_ratio = support / total_samples if total_samples > 0 else np.zeros_like(support)
    
    # 构建结果DataFrame
    result_df = pd.DataFrame({
        '认知层次': label_encoder.classes_,
        '精确率(Precision)': precision.round(4),
        '召回率(Recall)': recall.round(4),
        'F1值(F1-Score)': f1.round(4),
        '样本数': support,
        '样本占比(%)': (sample_ratio * 100).round(2)  # 百分比形式
    })
    
    return result_df

# -------------------------- 8. 主训练流程（含可视化） --------------------------
def train_and_evaluate_kv_model(processed_df: pd.DataFrame):
    """训练基于键值检索的认知层次分类模型（集成可视化）"""
    
    # 1. 准备数据
    queries = processed_df['query'].tolist()
    keys = processed_df['key'].tolist()
    labels = processed_df['value'].tolist()
    question_ids = processed_df['question_id'].tolist()

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    print(f"🔖 认知层次编码：{dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    # 初始化可视化工具
    visualizer = Visualizer(label_encoder)
    # 单独保存LabelEncoder
    save_label_encoder(label_encoder, 'label_encoder.pkl')

    # 2. 加载BERT分词器
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    # 3. 五折交叉验证（按问题id分组）
    unique_question_ids = list(set(question_ids))
    
    # 为每个问题分配代表性标签（用于分层抽样）
    question_label_map = {}
    for qid in unique_question_ids:
        qid_mask = [qid == q for q in question_ids]
        if sum(qid_mask) > 0:
            qid_labels = [labels[i] for i, mask in enumerate(qid_mask) if mask]
            question_label_map[qid] = qid_labels[0]  # 取第一个标签作为代表
    
    question_labels = [question_label_map[qid] for qid in unique_question_ids]
    question_label_encoded = label_encoder.transform(question_labels)

    kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    all_preds = []
    all_true = []
    all_question_ids = []
    best_fold_idx = -1
    best_fold_acc = 0.0
    best_fold_details = None  # 存储最优折的详细性能

    for fold, (train_qidx, val_qidx) in enumerate(kf.split(unique_question_ids, question_label_encoded), 1):
        print(f"\n{'='*50}")
        print(f"第{fold}折交叉验证")
        print(f"{'='*50}")

        # 获取训练集和验证集的问题id
        train_question_ids = [unique_question_ids[i] for i in train_qidx]
        val_question_ids = [unique_question_ids[i] for i in val_qidx]

        # 按问题id划分样本
        train_mask = [qid in train_question_ids for qid in question_ids]
        val_mask = [qid in val_question_ids for qid in question_ids]

        train_queries = [queries[i] for i in range(len(queries)) if train_mask[i]]
        train_keys = [keys[i] for i in range(len(keys)) if train_mask[i]]
        train_labels = [y_encoded[i] for i in range(len(y_encoded)) if train_mask[i]]

        val_queries = [queries[i] for i in range(len(queries)) if val_mask[i]]
        val_keys = [keys[i] for i in range(len(keys)) if val_mask[i]]
        val_labels = [y_encoded[i] for i in range(len(y_encoded)) if val_mask[i]]
        val_question_ids_fold = [question_ids[i] for i in range(len(question_ids)) if val_mask[i]]

        print(f"训练集：{len(train_queries)}个样本（{len(train_question_ids)}个问题）")
        print(f"验证集：{len(val_queries)}个样本（{len(val_question_ids)}个问题）")
        
        # 检查验证集是否有样本
        if len(val_queries) == 0:
            print("⚠️ 验证集为空，跳过该折")
            continue

        # 创建数据加载器
        train_data_loader = create_kv_data_loader(
            queries=train_queries,
            keys=train_keys,
            labels=train_labels,
            tokenizer=tokenizer,
            max_query_len=Config.MAX_QUERY_LEN,
            max_key_len=Config.MAX_KEY_LEN,
            batch_size=Config.BATCH_SIZE,
            shuffle=True
        )
        
        val_data_loader = create_kv_data_loader(
            queries=val_queries,
            keys=val_keys,
            labels=val_labels,
            tokenizer=tokenizer,
            max_query_len=Config.MAX_QUERY_LEN,
            max_key_len=Config.MAX_KEY_LEN,
            batch_size=Config.BATCH_SIZE,
            shuffle=False
        )

        # 初始化模型
        model = KnowledgeAwareCognitiveClassifier(
            bert_model_name=Config.BERT_MODEL,
            num_classes=num_classes,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)

        # 定义损失函数和优化器
        loss_fn = nn.CrossEntropyLoss().to(Config.DEVICE)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            eps=1e-8
        )

        # 学习率调度器
        total_steps = len(train_data_loader) * Config.NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练过程记录
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 训练epochs
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\n📌 Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            
            # 训练
            train_loss, train_acc = train_kv_epoch(
                model=model,
                data_loader=train_data_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=Config.DEVICE
            )
            
            # 评估（不返回特征）
            val_result = eval_kv_model(
                model=model,
                data_loader=val_data_loader,
                loss_fn=loss_fn,
                device=Config.DEVICE,
                label_encoder=label_encoder,
                return_features=False
            )
            
            val_loss = val_result['loss']
            val_acc = val_result['accuracy']

            # 记录
            train_losses.append(train_loss)
            train_accs.append(train_acc.item())
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f}")

        # 最终评估（返回特征用于TSNE）
        final_val_result = eval_kv_model(
            model=model,
            data_loader=val_data_loader,
            loss_fn=loss_fn,
            device=Config.DEVICE,
            label_encoder=label_encoder,
            return_features=True
        )

        # 保存该折结果
        fold_result = {
            'fold_idx': fold,
            'train_loss': train_losses,
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs,
            'val_accuracy': final_val_result['accuracy'],
            'val_loss_final': final_val_result['loss'],
            'val_predictions': final_val_result['predictions'],
            'val_labels': final_val_result['labels'],
            'val_features': final_val_result['features'],
            'classification_report': final_val_result['classification_report']
        }
        fold_results.append(fold_result)

        # 更新最优折
        if final_val_result['accuracy'] > best_fold_acc:
            best_fold_acc = final_val_result['accuracy']
            best_fold_idx = fold
            # 计算最优折的类别级性能
            best_fold_details = get_class_level_performance(
                true_labels=final_val_result['labels'],
                pred_labels=final_val_result['predictions'],
                label_encoder=label_encoder
            )
            # 保存最优模型
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                label_encoder=label_encoder,
                epoch=Config.NUM_EPOCHS,
                val_acc=best_fold_acc,
                filepath=f"best_fold_{fold}_model.pt"
            )

        # 可视化该折结果
        visualizer.plot_loss_curve(train_losses, val_losses, fold)
        visualizer.plot_accuracy_curve(train_accs, val_accs, fold)
        visualizer.plot_class_accuracy(final_val_result['labels'], final_val_result['predictions'], fold)
        visualizer.plot_confusion_matrix(final_val_result['labels'], final_val_result['predictions'], fold)
        visualizer.plot_attention_weights(model, val_data_loader, Config.DEVICE, fold)
        if final_val_result['features'] is not None:
            visualizer.plot_tsne_features(final_val_result['features'], final_val_result['labels'], fold)

        # 收集所有预测结果
        all_preds.extend(final_val_result['predictions'])
        all_true.extend(final_val_result['labels'])
        all_question_ids.extend(val_question_ids_fold)

    # 输出跨折汇总指标
    calculate_fold_metrics_summary(fold_results)

    # ====================== 【核心修改：输出最优折详细性能】 ======================
    print(f"\n" + "="*80)
    print(f"📊 最优交叉验证折：第 {best_fold_idx} 折 | 最优验证准确率：{best_fold_acc:.4f}")
    print("="*80)
    print("🎯 最优折 - 各认知层次分类性能指标（精确率、召回率、F1值、样本占比）")
    print("-"*80)
    print(best_fold_details.to_string(index=False, float_format='%.4f'))
    print("-"*80)
    
    # 保存到CSV
    best_fold_details.to_csv('最优折_认知层次分类性能报告.csv', index=False, encoding='utf-8-sig')
    print("✅ 已将最优折性能数据保存至：最优折_认知层次分类性能报告.csv")
    # ============================================================================

    return fold_results, best_fold_details

# -------------------------- 9. 主函数 --------------------------
def main():
    """主执行函数"""
    try:
        # 1. 加载数据
        annotation_df = load_annotation_data(Config.ANNOTATION_PATH)
        solution_df = load_solution_data(Config.SOLUTION_PATH)
        
        # 2. 数据重构
        processed_df = reconstruct_data_for_kv_retrieval(annotation_df, solution_df)
        
        # 3. 训练模型并评估
        fold_results, best_fold_details = train_and_evaluate_kv_model(processed_df)
        
        print("\n🎉 训练完成！所有结果已保存至 visualizations6 文件夹")
        
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")
        raise

if __name__ == "__main__":
    main()
