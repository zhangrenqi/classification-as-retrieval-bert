import os
import json
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# sklearn 相关
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 模型
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm import LGBMClassifier

# 分词
from transformers import BertTokenizer

# 忽略警告
warnings.filterwarnings('ignore')


# -------------------------- 1. 配置参数与可视化设置 --------------------------
class Config:
    ANNOTATION_PATH = "question_knowledge_info_（标注）.txt"
    SOLUTION_PATH = "question_with_knowledge_solutions.json"
    BERT_MODEL = "bert-base-chinese"
    SEED = 42
    N_FOLDS = 5
    SAVE_DIR = "baseline_visualizations_bert"
    DPI = 300


os.makedirs(Config.SAVE_DIR, exist_ok=True)

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
        cm = confusion_matrix(y_true, y_pred)
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

        bars = plt.bar(names, values, color=self.colors)
        plt.xlabel('Cognitive Level', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Class-wise Accuracy - {model_name}', fontsize=14)
        plt.ylim(0, 1.05)

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
    try:
        anno_df = pd.read_csv(Config.ANNOTATION_PATH, encoding='utf-8')
    except:
        anno_df = pd.read_csv(Config.ANNOTATION_PATH, encoding='gbk')
    anno_df.columns = [col.strip() for col in anno_df.columns]
    anno_df = anno_df.drop_duplicates(subset=['问题id', '知识点名称'])

    with open(Config.SOLUTION_PATH, 'r', encoding='utf-8') as f:
        sol_data = json.load(f)
    sol_df = pd.DataFrame(sol_data['data_list'])[['question_id', 'solution_steps']]
    sol_df = sol_df.rename(columns={'question_id': '问题id'}).drop_duplicates(subset=['问题id'])

    anno_df['问题id'] = anno_df['问题id'].astype(str).str.strip()
    sol_df['问题id'] = sol_df['问题id'].astype(str).str.strip()

    df = pd.merge(anno_df, sol_df, on='问题id', how='inner')
    df = df[df['solution_steps'].notna() & df['认知目标层次'].notna()]

    print("⏳ 正在进行 BERT 分词 (用于对齐深度学习模型的输入)...")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL)

    features = []
    labels = []
    question_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        q_tokens = tokenizer.tokenize(row['solution_steps'])[:256]
        k_tokens = tokenizer.tokenize(row['知识点名称'])[:64]
        text = " ".join(q_tokens) + " [SEP] " + " ".join(k_tokens)

        features.append(text)
        labels.append(row['认知目标层次'])
        question_ids.append(row['问题id'])

    return features, labels, question_ids


# -------------------------- 4. 模型 --------------------------
def get_models(random_state=42):
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
        "Naive_Bayes": MultinomialNB(),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1)
    }


# -------------------------- 主函数 --------------------------
def main():
    print("=" * 60)
    print("  Baseline Models Evaluation (Aligned with BERT Experiment)")
    print("=" * 60)

    texts, labels, q_ids = load_and_process_data()

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f"✅ 类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    visualizer = BaselineVisualizer(le, Config.SAVE_DIR)

    print("📊 提取 TF-IDF 特征...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3)
    X = tfidf.fit_transform(texts)

    q_ids = np.array(q_ids)
    unique_qids = np.unique(q_ids)
    q_id_to_label = {qid: lab for qid, lab in zip(q_ids, y)}
    unique_labels = [q_id_to_label[qid] for qid in unique_qids]

    kf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    models = get_models(Config.SEED)
    final_results = []

    for name, model in models.items():
        print(f"\n🚀 正在评估模型: {name}")
        fold_accs = []
        fold_macro = []
        fold_weighted = []
        all_true = []
        all_pred = []

        for fold, (train_idx_q, val_idx_q) in enumerate(kf.split(unique_qids, unique_labels), 1):
            train_qids = unique_qids[train_idx_q]
            val_qids = unique_qids[val_idx_q]

            train_mask = np.isin(q_ids, train_qids)
            val_mask = np.isin(q_ids, val_qids)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            macro = f1_score(y_val, preds, average='macro')
            weighted = f1_score(y_val, preds, average='weighted')

            fold_accs.append(acc)
            fold_macro.append(macro)
            fold_weighted.append(weighted)
            all_true.extend(y_val)
            all_pred.extend(preds)

        # 计算均值和标准差
        acc_mean = np.mean(fold_accs)
        acc_std = np.std(fold_accs)
        macro_mean = np.mean(fold_macro)
        macro_std = np.std(fold_macro)
        weighted_mean = np.mean(fold_weighted)
        weighted_std = np.std(fold_weighted)

        # 【按你要求的格式输出】
        print(f"\n【{name} 5折交叉验证结果】")
        print(f"准确率:    {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"Macro-F1:  {macro_mean:.4f} ± {macro_std:.4f}")
        print(f"Weighted-F1: {weighted_mean:.4f} ± {weighted_std:.4f}")

        print(f"\n📊 详细分类报告")
        print(classification_report(all_true, all_pred, target_names=le.classes_, digits=4))

        # 可视化
        visualizer.plot_confusion_matrix(all_true, all_pred, name)
        visualizer.plot_class_accuracy(all_true, all_pred, name)

        final_results.append({
            "Model": name,
            "Acc_Mean": acc_mean,
            "Acc_Std": acc_std,
            "Macro_F1_Mean": macro_mean,
            "Macro_F1_Std": macro_std,
            "Weighted_F1_Mean": weighted_mean,
            "Weighted_F1_Std": weighted_std
        })

    # 最终汇总表格
    print("\n" + "=" * 80)
    print("🏆 最终基线模型对比表")
    print("=" * 80)
    for res in final_results:
        print(f"{res['Model']:18} | Acc: {res['Acc_Mean']:.4f}±{res['Acc_Std']:.4f} | "
              f"Macro-F1: {res['Macro_F1_Mean']:.4f}±{res['Macro_F1_Std']:.4f} | "
              f"Weighted-F1: {res['Weighted_F1_Mean']:.4f}±{res['Weighted_F1_Std']:.4f}")

    print("=" * 80)
    print(f"✅ 所有结果已保存到: {Config.SAVE_DIR}")


if __name__ == "__main__":
    main()