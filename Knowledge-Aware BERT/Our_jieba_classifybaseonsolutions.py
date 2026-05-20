
#添加可视化要求：
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
from collections import defaultdict
import pickle
# 可视化相关库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
# 新增jieba分词库
import jieba
# 新增提速相关库
import multiprocessing
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import gc  # 垃圾回收

# 忽略警告
warnings.filterwarnings("ignore")

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-cache'
os.environ['MATPLOTLIB_VERBOSITY'] = 'quiet'
# 新增CUDA优化环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 非阻塞CUDA启动
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count() // 2)  # 限制CPU线程数避免竞争

import matplotlib
matplotlib.rcParams['backend'] = 'Agg'  # 无界面模式
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 关闭所有日志
import logging
logging.getLogger('matplotlib').setLevel(logging.FATAL)
logging.getLogger('PIL').setLevel(logging.FATAL)
logging.getLogger('transformers').setLevel(logging.ERROR)  # 关闭transformers冗余日志

# 创建可视化结果保存文件夹
os.makedirs('visualizations_baseonsolutions_jieba', exist_ok=True)

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = False  # 关闭确定性，提升速度
    torch.backends.cudnn.benchmark = True      # 开启自动调优，适配最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速矩阵运算
    torch.backends.cudnn.allow_tf32 = True

# -------------------------- 1. 配置参数（新增提速相关） --------------------------
class Config:
    # 数据路径
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"  # 标注数据集
    SOLUTION_PATH = "question_with_knowledge_solutions.json"  # 解题步骤数据集
    # BERT配置
    BERT_MODEL = "bert-base-chinese"  # 中文BERT模型
    MAX_QUERY_LEN = 512  # 解题过程最大长度
    MAX_KEY_LEN = 128     # 知识点最大长度
    # 模型训练配置
    BATCH_SIZE = 32  # 优化：增大批次（根据GPU显存调整，原16→32）
    LEARNING_RATE = 2e-5   
    NUM_EPOCHS = 10
    HIDDEN_DIM = 256  # 分类头隐藏层维度
    # 交叉验证配置
    N_FOLDS = 5  # 五折交叉验证
    # 设备配置 - 强制校验CUDA，无CUDA则报错
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cpu":
        raise RuntimeError("当前环境无CUDA可用！请确保GPU已安装并配置CUDA环境")
    # 可视化配置（仅保留2D TSNE参数）
    TSNE_PERPLEXITY = 30  # TSNE降维参数
    TSNE_max_iter = 1000   # TSNE迭代次数
    PLOT_DPI = 300         # 图片分辨率
    # 新增提速配置
    NUM_WORKERS = multiprocessing.cpu_count()  # 数据加载线程数（满核心）
    PIN_MEMORY = True                          # 固定内存加速CUDA传输
    MIXED_PRECISION = True                     # 混合精度训练
    GRADIENT_ACCUMULATION_STEPS = 1            # 梯度累积（显存不足时设>1）
    CACHE_TOKENS = True                        # 缓存分词结果，避免重复计算

# -------------------------- 新增：jieba分词适配BERT的工具函数（优化） --------------------------
def jieba_tokenize_for_bert(text, tokenizer, max_length):
    """
    使用jieba分词替换BERT原生分词，并适配BERT输入格式（优化：减少冗余操作）
    :param text: 原始文本
    :param tokenizer: BERT的tokenizer（用于词表映射和特殊标记）
    :param max_length: 最大长度
    :return: 符合BERT格式的input_ids, attention_mask
    """
    # 优化：提前过滤空文本
    text = str(text).strip()
    if not text:
        bert_tokens = ['[CLS]', '[SEP]']
    else:
        # 1. jieba分词（精确模式）
        tokens = jieba.lcut(text, cut_all=False)
        
        # 2. 将jieba分词结果映射到BERT词表（优化：批量判断，减少循环）
        bert_tokens = []
        for token in tokens:
            if token in tokenizer.vocab:
                bert_tokens.append(token)
            else:
                bert_tokens.extend(list(token))
        
        # 3. 添加BERT特殊标记（[CLS]开头，[SEP]结尾）
        bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
    
    # 4. 截断超长token（优化：提前判断，减少切片操作）
    if len(bert_tokens) > max_length:
        bert_tokens = bert_tokens[:max_length]
    
    # 5. 转换为input_ids，并填充至max_length（优化：批量操作）
    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    
    # 6. 生成attention_mask（优化：列表生成式提速）
    attention_mask = [1] * len(bert_tokens) + [0] * padding_length
    
    # 优化：直接返回numpy数组，后续统一转张量（减少张量创建开销）
    return np.array(input_ids, dtype=np.int64), np.array(attention_mask, dtype=np.int64)

# -------------------------- 新增：缓存分词结果的工具函数 --------------------------
def preprocess_and_cache_tokens(processed_df, tokenizer, config):
    """预处理并缓存所有文本的分词结果（避免每个epoch重复分词）"""
    cache_path = "token_cache.pkl"
    if config.CACHE_TOKENS and os.path.exists(cache_path):
        print("✅ 加载缓存的分词结果...")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['query_ids'], cache_data['query_masks'], cache_data['key_ids'], cache_data['key_masks']
    
    print("🔄 预处理并缓存分词结果...")
    # 批量处理分词（优化：使用列表推导式提速）
    query_ids = []
    query_masks = []
    key_ids = []
    key_masks = []
    
    # 多进程分词（可选，进一步提速）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        # 处理query
        query_futures = [executor.submit(jieba_tokenize_for_bert, text, tokenizer, config.MAX_QUERY_LEN) 
                         for text in processed_df['query'].tolist()]
        for future in tqdm(query_futures, desc="处理Query分词"):
            ids, mask = future.result()
            query_ids.append(ids)
            query_masks.append(mask)
        
        # 处理key
        key_futures = [executor.submit(jieba_tokenize_for_bert, text, tokenizer, config.MAX_KEY_LEN) 
                       for text in processed_df['key'].tolist()]
        for future in tqdm(key_futures, desc="处理Key分词"):
            ids, mask = future.result()
            key_ids.append(ids)
            key_masks.append(mask)
    
    # 转换为numpy数组（优化：批量转换，减少内存碎片）
    query_ids = np.vstack(query_ids)
    query_masks = np.vstack(query_masks)
    key_ids = np.vstack(key_ids)
    key_masks = np.vstack(key_masks)
    
    # 缓存结果
    if config.CACHE_TOKENS:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'query_ids': query_ids,
                'query_masks': query_masks,
                'key_ids': key_ids,
                'key_masks': key_masks
            }, f)
    
    return query_ids, query_masks, key_ids, key_masks

# -------------------------- 2. 数据加载与预处理（无逻辑修改） --------------------------
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

def load_solution_data(path: str) -> pd.DataFrame:
    """加载解题步骤数据（按问题id唯一存储）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"解题步骤文件不存在：{path}")

    with open(path, 'r', encoding='utf-8') as f:
        solution_json = json.load(f)

    # 转换为DataFrame（每个问题id只保留一条记录，含完整解题步骤）
    solution_df = pd.DataFrame(solution_json['data_list'])[[
        'question_id', 'solution_steps', 'knowledge_points'
    ]].rename(columns={
        'question_id': '问题id',
        'knowledge_points': '知识点列表'  # 保留原始知识点列表用于校验
    })

    # 确保问题id唯一（一个问题对应一个解题步骤）
    solution_df = solution_df.drop_duplicates(subset=['问题id'], keep='first')
    print(f"✅ 加载解题步骤数据：{len(solution_df)}个唯一问题（每个问题一个解题步骤）")
    return solution_df

def reconstruct_data_for_kv_retrieval(annotation_df: pd.DataFrame, solution_df: pd.DataFrame) -> pd.DataFrame:
    """
    重构数据格式，适应键值检索需求
    Query: 解题过程
    Key: 知识点
    Value: 认知目标层次
    """
    # 统一字符串格式
    annotation_df['问题id'] = annotation_df['问题id'].astype(str).str.strip()
    annotation_df['知识点名称'] = annotation_df['知识点名称'].astype(str).str.strip()
    solution_df['问题id'] = solution_df['问题id'].astype(str).str.strip()

    # 按问题id合并
    merged_df = pd.merge(
        annotation_df,
        solution_df[['问题id', 'solution_steps']],
        on=['问题id'],
        how='inner'
    ).rename(columns={'solution_steps': '解题步骤'})

    # 过滤无效数据
    merged_df = merged_df[
        (merged_df['解题步骤'].notna()) &
        (merged_df['解题步骤'].str.strip() != '') &
        (merged_df['认知目标层次'].notna()) &
        (merged_df['认知目标层次'].str.strip() != '')
    ]

    processed_data = []
    
    # 按问题分组，重构数据格式
    for question_id, group in merged_df.groupby('问题id'):
        # 解题过程作为query（相同的查询内容）
        solution_text = group['解题步骤'].iloc[0]  # 每个问题只有一个解题过程
        
        # 该问题的每个知识点作为独立的key-value对
        for _, row in group.iterrows():
            knowledge_point = f"{row['知识点名称']}"
            cognitive_level = row['认知目标层次']
            
            processed_data.append({
                'question_id': question_id,
                'query': solution_text,           # Query: 解题过程
                'key': knowledge_point,           # Key: 知识点
                'value': cognitive_level,         # Value: 解题过程+知识点
                'fusion_text': f"解题过程:{solution_text}；知识点:{knowledge_point}"
            })
    
    processed_df = pd.DataFrame(processed_data)
    
    # 统计信息
    print(f"✅ 重构后数据量：{len(processed_df)}条记录")
    print(f"📊 涉及问题数量：{processed_df['question_id'].nunique()}个")
    print(f"📊 平均每个问题包含知识点：{len(processed_df) / processed_df['question_id'].nunique():.2f}个")
    print("认知层次分布：")
    print(processed_df['value'].value_counts().to_string())
    
    return processed_df

# -------------------------- 3. 数据集与数据加载器（优化提速） --------------------------
class KVCognitiveDataset(Dataset):
    """基于键值检索的认知层次分类数据集（优化：直接加载缓存的numpy数组）"""
    def __init__(self, query_ids, query_masks, key_ids, key_masks, labels):
        self.query_ids = query_ids  # 预缓存的input_ids（numpy）
        self.query_masks = query_masks  # 预缓存的attention_mask（numpy）
        self.key_ids = key_ids
        self.key_masks = key_masks
        self.labels = labels    # 认知层次标签列表

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        # 优化：直接从numpy数组取数，减少转换开销
        query_input_ids = torch.from_numpy(self.query_ids[idx])
        query_attention_mask = torch.from_numpy(self.query_masks[idx])
        key_input_ids = torch.from_numpy(self.key_ids[idx])
        key_attention_mask = torch.from_numpy(self.key_masks[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'key_input_ids': key_input_ids,
            'key_attention_mask': key_attention_mask,
            'labels': label,
            # 注：若需文本可视化，需额外缓存query_text/key_text，此处为提速暂省（可根据需求加回）
            'query_text': "",
            'key_text': ""
        }

def create_kv_data_loader(query_ids, query_masks, key_ids, key_masks, labels, batch_size, shuffle=True):
    """创建键值检索数据加载器（优化：使用预缓存数据）"""
    ds = KVCognitiveDataset(
        query_ids=query_ids,
        query_masks=query_masks,
        key_ids=key_ids,
        key_masks=key_masks,
        labels=labels
    )

    # 优化：调整DataLoader参数提速
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
        prefetch_factor=2,  # 预取2个批次，减少等待
        persistent_workers=True  # 保持worker进程，避免重复创建
    )

# -------------------------- 4. 模型定义（无逻辑修改，仅优化显存） --------------------------
class KnowledgeAwareCognitiveClassifier(nn.Module):
    """基于键值检索思路的认知层次分类模型（支持特征提取）"""
    
    def __init__(self, bert_model_name, num_classes, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 注意力层用于query-key交互
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size * 2)
    
    def forward(self, query_input_ids, query_attention_mask, key_input_ids, key_attention_mask, return_features=False):
        """前向传播（支持返回特征用于可视化）"""
        # 编码解题过程 (query)
        query_outputs = self.bert(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embeddings = query_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 编码知识点 (key)
        key_outputs = self.bert(
            input_ids=key_input_ids,
            attention_mask=key_attention_mask
        )
        key_embeddings = key_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 计算query和key的注意力交互
        attended_output, attention_weights = self.attention_layer(
            query=query_embeddings,
            key=key_embeddings,
            value=key_embeddings,
            key_padding_mask=~key_attention_mask.bool() if key_attention_mask is not None else None
        )
        
        # 取[CLS]标记的特征进行融合
        query_cls = query_embeddings[:, 0, :]  # 解题过程的[CLS]特征
        attended_cls = attended_output[:, 0, :]  # 注意力加权的特征
        
        # 特征融合
        fused_features = torch.cat([query_cls, attended_cls], dim=1)
        fused_features = self.layer_norm(fused_features)
        
        # 分类预测
        logits = self.classifier(fused_features)
        
        # 若需要特征，则返回特征；否则返回logits和注意力权重
        if return_features:
            return logits, attention_weights, fused_features  # fused_features用于TSNE
        return logits, attention_weights

# -------------------------- 5. 保存和加载辅助函数（优化显存） --------------------------
def save_checkpoint(model, optimizer, label_encoder, epoch, val_acc, filepath):
    # 优化：不移动模型到CPU，直接保存（减少数据传输）
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'label_encoder_classes': label_encoder.classes_,
        'epoch': epoch,
        'val_acc': val_acc
    }
    # 优化：使用torch.save的高效格式
    torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
    print(f"✅ Checkpoint保存成功: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device=Config.DEVICE):
    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=True)
    except:
        print("⚠️ 使用weights_only=False加载checkpoint（确保文件来源可信）")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # 确保加载后模型在CUDA上
    
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

# -------------------------- 6. 可视化工具类（无逻辑修改） --------------------------
class Visualizer:
    """实现5项可视化功能（删除3D TSNE）"""
    def __init__(self, label_encoder, save_dir='visualizations_baseonsolutions_jieba', dpi=Config.PLOT_DPI):
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
        """5. 注意力权重可视化（展示query与key的交互）"""
        model.eval()
        samples_plotted = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if samples_plotted >= num_samples:
                    break
                    
                # 取第一个样本并移到CUDA
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

# -------------------------- 7. 训练与评估函数（核心提速优化） --------------------------
def train_kv_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, scaler=None):
    """训练一个epoch（优化：混合精度+梯度累积+减少冗余操作）"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    step = 0

    for batch in tqdm(data_loader, desc="训练中"):
        step += 1
        # 移动到CUDA设备（非阻塞传输）
        query_input_ids = batch["query_input_ids"].to(device, non_blocking=True)
        query_attention_mask = batch["query_attention_mask"].to(device, non_blocking=True)
        key_input_ids = batch["key_input_ids"].to(device, non_blocking=True)
        key_attention_mask = batch["key_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # 混合精度训练
        with autocast(enabled=Config.MIXED_PRECISION):
            # 前向传播
            outputs, _ = model(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                key_input_ids=key_input_ids,
                key_attention_mask=key_attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            # 梯度累积
            loss = loss / Config.GRADIENT_ACCUMULATION_STEPS

        # 统计正确预测数和损失
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS

        # 反向传播（混合精度）
        if Config.MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度裁剪+优化（仅在累积步数达标时）
        if step % Config.GRADIENT_ACCUMULATION_STEPS == 0:
            if Config.MIXED_PRECISION:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if Config.MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

    # 计算平均损失和准确率
    epoch_loss = total_loss / len(data_loader)
    epoch_acc = correct_predictions.double() / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc

def eval_kv_model(model, data_loader, loss_fn, device, label_encoder, return_features=False):
    """评估模型（支持返回特征用于TSNE）（优化：混合精度+批量处理）"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_features = []  # 存储BERT特征用于TSNE

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            # 移动到CUDA设备（非阻塞传输）
            query_input_ids = batch["query_input_ids"].to(device, non_blocking=True)
            query_attention_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            key_input_ids = batch["key_input_ids"].to(device, non_blocking=True)
            key_attention_mask = batch["key_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # 混合精度评估
            with autocast(enabled=Config.MIXED_PRECISION):
                # 前向传播（根据需要返回特征）
                if return_features:
                    outputs, _, features = model(
                        query_input_ids=query_input_ids,
                        query_attention_mask=query_attention_mask,
                        key_input_ids=key_input_ids,
                        key_attention_mask=key_attention_mask,
                        return_features=True
                    )
                    all_features.append(features.cpu().numpy())  # 特征移到CPU用于后续处理
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
            
            all_preds.extend(preds.cpu().numpy())  # 预测结果移到CPU
            all_labels.extend(labels.cpu().numpy())  # 标签移到CPU

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
    fold_losses = [r['val_loss'] for r in fold_results]
    fold_train_accs = [r['train_accuracy'] for r in fold_results]
    fold_train_losses = [r['train_loss'] for r in fold_results]
    
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

    # 严格按照要求的格式输出
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

# -------------------------- 8. 主训练流程（含可视化+提速优化） --------------------------
def train_and_evaluate_kv_model(processed_df: pd.DataFrame):
    """训练基于键值检索的认知层次分类模型（集成可视化+提速优化）"""
    
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
    visualizer = Visualizer(label_encoder, save_dir='visualizations_baseonsolutions_jieba')
    # 单独保存LabelEncoder
    save_label_encoder(label_encoder, 'label_encoder.pkl')

    # 2. 加载BERT分词器（仅用于词表映射，分词逻辑已替换为jieba）
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    # 优化：预缓存所有分词结果（避免重复计算）
    query_ids, query_masks, key_ids, key_masks = preprocess_and_cache_tokens(processed_df, tokenizer, Config)

    # 3. 五折交叉验证（按问题id分组）
    unique_question_ids = list(set(question_ids))
    
    # 为每个问题分配代表性标签（用于分层抽样）
    question_to_label = {}
    for qid, label in zip(question_ids, y_encoded):
        if qid not in question_to_label:
            question_to_label[qid] = label
    question_labels = [question_to_label[qid] for qid in unique_question_ids]
    
    # 初始化交叉验证
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold_idx, (train_q_idx, val_q_idx) in enumerate(skf.split(unique_question_ids, question_labels)):
        print(f"\n{'='*50} 开始训练第 {fold_idx+1}/{Config.N_FOLDS} 折 {'='*50}")
        
        # 按问题id划分训练/验证集
        train_question_ids = [unique_question_ids[i] for i in train_q_idx]
        val_question_ids = [unique_question_ids[i] for i in val_q_idx]
        
        # 映射到原始数据索引
        train_mask = processed_df['question_id'].isin(train_question_ids)
        val_mask = processed_df['question_id'].isin(val_question_ids)
        
        # 从缓存中提取训练/验证数据（优化：直接切片numpy数组）
        train_query_ids = query_ids[train_mask]
        train_query_masks = query_masks[train_mask]
        train_key_ids = key_ids[train_mask]
        train_key_masks = key_masks[train_mask]
        train_labels = y_encoded[train_mask]
        
        val_query_ids = query_ids[val_mask]
        val_query_masks = query_masks[val_mask]
        val_key_ids = key_ids[val_mask]
        val_key_masks = key_masks[val_mask]
        val_labels = y_encoded[val_mask]
        
        # 创建数据加载器
        train_loader = create_kv_data_loader(
            train_query_ids, train_query_masks, train_key_ids, train_key_masks, train_labels,
            batch_size=Config.BATCH_SIZE, shuffle=True
        )
        val_loader = create_kv_data_loader(
            val_query_ids, val_query_masks, val_key_ids, val_key_masks, val_labels,
            batch_size=Config.BATCH_SIZE*2, shuffle=False  # 验证集批次翻倍，提速
        )
        
        # 初始化模型（优化：使用torch.compile加速）
        model = KnowledgeAwareCognitiveClassifier(
            bert_model_name=Config.BERT_MODEL,
            num_classes=num_classes,
            hidden_dim=Config.HIDDEN_DIM
        ).to(Config.DEVICE)
        # 编译模型（PyTorch 2.0+特性，提速30%+）
        if torch.__version__ >= "2.0.0":
            model = torch.compile(model)
        
        # 初始化优化器和调度器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            eps=1e-8,
            weight_decay=0.01  # 权重衰减，防止过拟合
        )
        total_steps = len(train_loader) * Config.NUM_EPOCHS // Config.GRADIENT_ACCUMULATION_STEPS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10%预热
            num_training_steps=total_steps
        )
        loss_fn = nn.CrossEntropyLoss().to(Config.DEVICE)
        
        # 混合精度训练缩放器
        scaler = GradScaler() if Config.MIXED_PRECISION else None
        
        # 训练记录
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_val_acc = 0.0
        
        # 训练epochs
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print("-" * 30)
            
            # 训练
            train_loss, train_acc = train_kv_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, Config.DEVICE, scaler
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc.item())
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc.item():.4f}")
            
            # 评估
            val_results = eval_kv_model(
                model, val_loader, loss_fn, Config.DEVICE, label_encoder, return_features=(epoch==Config.NUM_EPOCHS-1)
            )
            val_losses.append(val_results['loss'])
            val_accs.append(val_results['accuracy'])
            print(f"验证损失: {val_results['loss']:.4f}, 验证准确率: {val_results['accuracy']:.4f}")
            print(f"分类报告:\n{val_results['classification_report']}")
            
            # 保存最佳模型
            if val_results['accuracy'] > best_val_acc:
                best_val_acc = val_results['accuracy']
                save_checkpoint(
                    model, optimizer, label_encoder, epoch, best_val_acc,
                    f"best_model_fold_{fold_idx+1}.pth"
                )
        
        # 可视化
        visualizer.plot_loss_curve(train_losses, val_losses, fold_idx+1)
        visualizer.plot_accuracy_curve(train_accs, val_accs, fold_idx+1)
        # 重新评估获取完整结果（含特征）
        final_val_results = eval_kv_model(
            model, val_loader, loss_fn, Config.DEVICE, label_encoder, return_features=True
        )
        visualizer.plot_class_accuracy(final_val_results['labels'], final_val_results['predictions'], fold_idx+1)
        visualizer.plot_confusion_matrix(final_val_results['labels'], final_val_results['predictions'], fold_idx+1)
        # visualizer.plot_attention_weights(model, val_loader, Config.DEVICE, fold_idx+1)  # 可选，耗时较长
        visualizer.plot_tsne_features(final_val_results['features'], final_val_results['labels'], fold_idx+1)
        
        # 记录折结果
        fold_results.append({
            'train_loss': train_losses[-1],
            'train_accuracy': train_accs[-1],
            'val_loss': final_val_results['loss'],
            'val_accuracy': final_val_results['accuracy'],
            'val_predictions': final_val_results['predictions'],
            'val_labels': final_val_results['labels']
        })
        
        # 优化：清理显存，避免OOM
        del model, optimizer, scheduler, scaler
        torch.cuda.empty_cache()
        gc.collect()
    
    # 计算跨折指标
    calculate_fold_metrics_summary(fold_results)
    print("\n🎉 所有折训练完成！")

# -------------------------- 执行入口 --------------------------
if __name__ == "__main__":
    # 加载数据
    annotation_df = load_annotation_data(Config.ANNOTATION_PATH)
    solution_df = load_solution_data(Config.SOLUTION_PATH)
    processed_df = reconstruct_data_for_kv_retrieval(annotation_df, solution_df)
    
    # 开始训练
    train_and_evaluate_kv_model(processed_df)