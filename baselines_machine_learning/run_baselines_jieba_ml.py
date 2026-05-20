import os
import json
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jieba  # <--- 引入 Jieba 分词库

# sklearn 相关
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer#将文本转换为 TF-IDF 向量，特征维度由 TfidfVectorizer 根据语料自动确定
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 模型
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

# 忽略警告
warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数与可视化设置 --------------------------
class Config:
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"
    SOLUTION_PATH = "question_with_knowledge_solutions.json"
    # BERT_MODEL = "bert-base-chinese"  <--- 不再需要 BERT 模型配置
    SEED = 42
    N_FOLDS = 5  # 五折交叉验证
    SAVE_DIR = "baseline_visualizations_jieba"  # 修改保存路径以区分
    DPI = 300


# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# 设置中文字体（与原代码保持一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 2. 可视化工具类 --------------------------
class BaselineVisualizer:
    def __init__(self, label_encoder, save_dir):
        self.le = label_encoder
        self.classes = label_encoder.classes_
        self.save_dir = save_dir
        self.colors = sns.color_palette("husl", len(self.classes))

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """生成并保存混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path, dpi=Config.DPI)
        plt.close()

    def plot_class_accuracy(self, y_true, y_pred, model_name):
        """生成并保存各类别准确率柱状图"""
        class_acc = {}
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        for idx, cls_name in enumerate(self.classes):
            mask = (y_true == idx)
            if np.sum(mask) == 0:
                class_acc[cls_name] = 0.0
            else:
                class_acc[cls_name] = np.mean(y_pred[mask] == idx)

        plt.figure(figsize=(10, 6))
        names = list(class_acc.keys())
        values = list(class_acc.values())

        # 绘制柱状图
        bars = plt.bar(names, values, color=self.colors)
        plt.xlabel('Cognitive Level', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Class-wise Accuracy - {model_name}', fontsize=14)
        plt.ylim(0, 1.05)

        # 标注数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{model_name}_class_accuracy.png")
        plt.savefig(save_path, dpi=Config.DPI)
        plt.close()


# -------------------------- 3. 数据加载与预处理 --------------------------
def load_and_process_data():
    """使用 Jieba 分词加载和处理数据"""
    # 1. 加载标注
    try:
        anno_df = pd.read_csv(Config.ANNOTATION_PATH, encoding='utf-8')
    except:
        anno_df = pd.read_csv(Config.ANNOTATION_PATH, encoding='gbk')
    anno_df.columns = [col.strip() for col in anno_df.columns]
    anno_df = anno_df.drop_duplicates(subset=['问题id', '知识点名称'])

    # 2. 加载解题步骤
    with open(Config.SOLUTION_PATH, 'r', encoding='utf-8') as f:
        sol_data = json.load(f)
    sol_df = pd.DataFrame(sol_data['data_list'])[['question_id', 'solution_steps']]
    sol_df = sol_df.rename(columns={'question_id': '问题id'}).drop_duplicates(subset=['问题id'])

    # 3. 统一类型并合并
    anno_df['问题id'] = anno_df['问题id'].astype(str).str.strip()
    sol_df['问题id'] = sol_df['问题id'].astype(str).str.strip()

    df = pd.merge(anno_df, sol_df, on='问题id', how='inner')
    df = df[df['solution_steps'].notna() & df['认知目标层次'].notna()]

    print("⏳ 正在进行 Jieba 分词 (传统机器学习标准预处理)...")
    # 注意：这里不再实例化 BertTokenizer

    features = []
    labels = []
    question_ids = []

    # 预处理：分词并拼接
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # --- 修改部分开始 ---
        # 使用 Jieba 进行中文分词 (返回列表)
        q_words = jieba.lcut(str(row['solution_steps']))
        k_words = jieba.lcut(str(row['知识点名称']))

        # 简单的截断策略 (按词数截断，保持与之前大致相当的信息量)
        # BERT max_len 256 是字数，这里 256 个词包含的信息通常更多
        q_words = q_words[:256]
        k_words = k_words[:64]

        # 拼接策略：用空格连接列表中的词，这是 TfidfVectorizer 标准输入格式
        # 依然保留结构：解题步骤 + 知识点
        text = " ".join(q_words) + " " + " ".join(k_words)
        # --- 修改部分结束 ---

        features.append(text)
        labels.append(row['认知目标层次'])
        question_ids.append(row['问题id'])

    return features, labels, question_ids


# -------------------------- 4. 模型训练与评估主流程 --------------------------
def get_models(random_state=42):
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
        "Naive_Bayes": MultinomialNB(),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1)
    }


def main():
    print("=" * 60)
    print("  Baseline Models Evaluation (Jieba Tokenization)")
    print("=" * 60)

    # 1. 准备数据
    texts, labels, q_ids = load_and_process_data()

    # 2. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"✅ 类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 初始化可视化器
    visualizer = BaselineVisualizer(le, Config.SAVE_DIR)

    # 3. 特征向量化 (TF-IDF)
    print("📊 提取 TF-IDF 特征...")
    # TfidfVectorizer 默认使用空格分隔单词，这与我们上面 jieba " ".join() 的输出完美契合
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3)
    X = tfidf.fit_transform(texts)

    # 4. 交叉验证设置 (分层抽样)
    # 这里的逻辑与原代码一致：按问题ID进行划分，防止同一问题的不同知识点泄露
    q_ids = np.array(q_ids)
    unique_qids = np.unique(q_ids)

    # 获取每个问题的一个代表标签用于分层
    q_id_to_label = {qid: lab for qid, lab in zip(q_ids, y)}
    unique_labels = [q_id_to_label[qid] for qid in unique_qids]

    kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    models = get_models(Config.SEED)
    final_results = []

    # 5. 遍历模型
    for name, model in models.items():
        print(f"\n🚀 正在评估模型: {name}")

        all_true = []
        all_pred = []
        fold_accuracies = []

        # 交叉验证
        for fold, (train_idx_q, val_idx_q) in enumerate(kf.split(unique_qids, unique_labels), 1):
            train_qids = unique_qids[train_idx_q]
            val_qids = unique_qids[val_idx_q]

            # 生成 Mask
            train_mask = np.isin(q_ids, train_qids)
            val_mask = np.isin(q_ids, val_qids)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            if len(y_val) == 0: continue

            # 训练与预测
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            # 记录结果
            acc = accuracy_score(y_val, preds)
            fold_accuracies.append(acc)
            all_true.extend(y_val)
            all_pred.extend(preds)

            # 简单的进度打印
            # print(f"  Fold {fold}: Accuracy = {acc:.4f}")

        # --- 汇总评估 ---
        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        # 生成分类报告
        report = classification_report(all_true, all_pred, target_names=le.classes_)
        print(report)

        # 生成可视化图表
        print(f"🎨 保存可视化结果至 {Config.SAVE_DIR}...")
        visualizer.plot_confusion_matrix(all_true, all_pred, name)
        visualizer.plot_class_accuracy(all_true, all_pred, name)

        # 记录汇总数据
        report_dict = classification_report(all_true, all_pred, output_dict=True)
        macro_f1 = report_dict['macro avg']['f1-score']
        weighted_f1 = report_dict['weighted avg']['f1-score']
        
        # 按照你要求的格式输出指标
        print(f"准确率:    {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Macro-F1:  {macro_f1:.4f} ± {std_acc:.4f}")
        print(f"Weighted-F1: {weighted_f1:.4f} ± {std_acc:.4f}")

        final_results.append({
            "Model": name,
            "Accuracy": mean_acc,
            "Std": std_acc,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        })

    # 6. 最终对比表输出
    print("\n" + "=" * 80)
    print("🏆 FINAL BASELINE COMPARISON (Jieba)")
    print("=" * 80)
    results_df = pd.DataFrame(final_results).sort_values(by="Accuracy", ascending=False)
    print(results_df.to_string(index=False, float_format="%.4f"))
    print("=" * 80)
    print(f"✅ 所有结果已生成，请查看 '{Config.SAVE_DIR}' 文件夹内的图片。")


if __name__ == "__main__":
    main()