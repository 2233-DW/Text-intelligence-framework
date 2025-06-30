# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ================= 配置区 =================
INPUT_PATH = r"D:\SASanalysis\SAS_text\python_SAS\output_yuchuli\text_pairs_2.csv"
OUTPUT_DIR = r"D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tfidf_matrix_2.csv")  # 保持.csv扩展名

# === SAS/Excel兼容配置 ===
OUTPUT_SETTINGS = {
    'sep': ',',  # 明确指定逗号分隔（与旧版一致）
    'float_format': "%.9f",  # 保留9位小数（旧版格式）
    'encoding': 'utf_8_sig',  # 带BOM的UTF-8（兼容中文Excel）
    'index': True  # 保留文档ID列
}

# === 特征词处理配置 ===
FEATURE_SETTINGS = {
    'max_features': 1000,  # 保持旧版容量
    'token_pattern': r'(?u)\b\w+\b',  # 严格匹配单词
    'ngram_range': (1, 1),
    'norm': 'l2' # 仅一元语法# 必须显式设置
}


# =========================================

def clean_feature_names(features):
    """特征词标准化（严格模式）"""
    # 替换所有非字母数字字符为下划线
    return [re.sub(r'[^\w]', '_', f) for f in features]


def main():
    try:
        # === 数据加载 ===
        print("[1/4] 读取输入文件...")
        df = pd.read_csv(INPUT_PATH, encoding='utf_8_sig')

        # === 文本合并 ===
        all_texts = pd.concat([df['draft_clean'], df['final_clean']], ignore_index=True)
        print(f"[2/4] 合并完成，文档总数：{len(all_texts)}")

        # === 核心建模 ===
        print("[3/4] 计算TF-IDF矩阵...")
        tfidf = TfidfVectorizer(**FEATURE_SETTINGS)
        tfidf_matrix = tfidf.fit_transform(all_texts)

        # === 格式标准化 ===
        print("[4/4] 执行格式处理...")
        # 特征词清洗
        features = clean_feature_names(tfidf.get_feature_names_out())
        # 生成文档ID（与旧版完全一致）
        doc_ids = [f"doc{i + 1}" for i in range(len(all_texts))]

        # 构建DataFrame
        df_matrix = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=doc_ids,
            columns=features
        )

        # === 保存结果 ===
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_matrix.to_csv(OUTPUT_FILE, **OUTPUT_SETTINGS)

        print(f"✅ 成功生成兼容旧版格式的文件：{OUTPUT_FILE}")
        print(f"特征维度：{df_matrix.shape[1]} | 文档数量：{df_matrix.shape[0]}")

    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        print("应急处理：")
        print("1. 检查输入文件是否包含draft_clean/final_clean列")
        print("2. 确认输出目录权限（右键以管理员身份运行脚本）")


if __name__ == "__main__":
    main()