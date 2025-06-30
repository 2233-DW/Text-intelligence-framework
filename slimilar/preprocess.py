# -*- coding: utf-8 -*-
"""
文本预处理流程 v2.3
优化点：修复变量作用域问题 + 增强质量检查
"""

import os
import re
import jieba
import pandas as pd
from typing import Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
# ================= 配置区 =================
CUSTOM_DICT_PATH = r"D:\SASanalysis\SAS_text\comnew_dict.txt"
STOPWORDS_PATH = r"D:\SASanalysis\SAS_text\stopwords.txt"
DRAFT_PATH = r"D:\SASanalysis\SAS_text\head.txt"
FINAL_PATH = r"D:\SASanalysis\SAS_text\lastx_04.txt"
OUTPUT_PATH = r"D:\SASanalysis\SAS_text\python_SAS\output_yuchuli\text_pairs_2.csv"
DOC_ID_PREFIX = "P001"


# =========================================

def load_custom_dict(path: str) -> None:
    """加载自定义词典"""
    try:
        if os.path.exists(path):
            jieba.load_userdict(path)
            print(f"✅ 自定义词典加载成功：{os.path.basename(path)}")
        else:
            print(f"⚠️ 警告：自定义词典文件 {path} 不存在")
    except Exception as e:
        print(f"❌ 词典加载异常：{str(e)}")


def load_stopwords(path: str) -> Set[str]:
    """加载停用词表（增强编码兼容性）"""
    stopwords = set()
    encodings = ['utf-8', 'gbk', 'gb2312', 'big5']

    if os.path.exists(path):
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    stopwords = {line.strip() for line in f if line.strip()}
                print(f"✅ 停用词加载成功：{len(stopwords)} 个（编码：{encoding}）")
                return stopwords
            except UnicodeDecodeError:
                continue
        print("⚠️ 警告：未能识别停用词文件编码")
    else:
        print("⚠️ 警告：停用词文件不存在，使用空集合")
    return stopwords
    # 加强停用词加载验证

def process_file(file_path: str, stopwords: Set[str]) -> Tuple[str, str]:
    """
    处理单个文件
    返回：(原始文本, 清洗后文本)
    """# ==== 文件读取 ====
    raw_text = ""
    for encoding in ['utf-8', 'gbk', 'ansi']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                raw_text = f.read()
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ 文件读取失败：{file_path} - {str(e)}")
            return "", ""
    # ==== 文本清洗 ====
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", raw_text)
    cleaned_text = re.sub(r'\b\d+\b', ' ', cleaned_text)
    words = jieba.lcut(cleaned_text)
    # 过滤处理（增加中间输出）
    filtered = []
    for w in words:
        w = w.strip()
        if not w: continue
        if w in stopwords:
            # print(f"过滤停用词：{w}")  # 调试时启用
            continue
        if len(w) > 1 and not w.isdigit():
            filtered.append(w)
    print(f"正在处理：{os.path.basename(file_path)} | 使用停用词数量：{len(stopwords)}")
    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        token_pattern=r'(?u)\b\w[\w-]*\w\b',  # 增强中文分词匹配
        max_df=0.85  # 过滤高频词
    )
    # 读取原始内容
    raw_text = ""
    for encoding in ['utf-8', 'gbk', 'ansi']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                raw_text = f.read()
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ 文件读取失败：{file_path} - {str(e)}")
            return "", ""

    # 文本清洗流程
    cleaned_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", " ", raw_text)
    cleaned_text = re.sub(r'\b\d+\b', ' ', cleaned_text)

    # 精确模式分词
    words = jieba.lcut(cleaned_text)

    # 过滤处理
    filtered = [w.strip() for w in words
                if w.strip()
                and len(w.strip()) > 1
                and w not in stopwords
                and not w.isdigit()]

    return raw_text, " ".join(filtered)


def main() -> pd.DataFrame:
    """主处理流程（返回DataFrame用于调试）"""
    # ==== 初始化阶段 ====
    print("\n" + "=" * 30 + " 初始化配置 " + "=" * 30)
    jieba.initialize()
    load_custom_dict(CUSTOM_DICT_PATH)
    stopwords = load_stopwords(STOPWORDS_PATH)

    # ==== 数据处理阶段 ====
    print("\n" + "=" * 30 + " 处理文档 " + "=" * 30)
    draft_raw, draft_clean = process_file(DRAFT_PATH, stopwords)
    final_raw, final_clean = process_file(FINAL_PATH, stopwords)

    # ==== 数据验证阶段 ====
    print("\n" + "=" * 30 + " 质量检查 " + "=" * 30)
    if not draft_clean:
        raise ValueError("初稿清洗结果为空，请检查输入文件或分词设置")
    if not final_clean:
        raise ValueError("终稿清洗结果为空，请检查输入文件或分词设置")
        # 新增停用词校验
    sample_words = ['的', '是', '在', '要']  # 示例停用词
    draft_check = [w for w in sample_words if w in draft_clean.split()]
    final_check = [w for w in sample_words if w in final_clean.split()]
    print(f"残留停用词检测：初稿{len(draft_check)}个，终稿{len(final_check)}个")
    # 新增质量检查输出
    print("\n清洗结果预览：")
    print(f"初稿样例：{draft_clean[:100]}...")
    print(f"终稿样例：{final_clean[:100]}...")
    print(f"\n初稿有效词数：{len(draft_clean.split())}")
    print(f"终稿有效词数：{len(final_clean.split())}")
    # ==== 数据验证阶段 ====
    print("\n" + "=" * 30 + " 质量检查 " + "=" * 30)

    # ==== 结果输出阶段 ====
    print("\n" + "=" * 30 + " 保存结果 " + "=" * 30)
    df = pd.DataFrame({
        "doc_id": [DOC_ID_PREFIX],
        "draft": [draft_raw],
        "final": [final_raw],
        "draft_clean": [draft_clean],
        "final_clean": [final_clean]
    })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf_8_sig')
        print(f"✅ 文件保存成功：{OUTPUT_PATH}")
        return df
    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")
        raise


def test_segmentation():
    """分词测试函数"""
    test_cases = [
        "王船山的经世致用思想对现代经济发展有重要启示",
        "船山思想中的区域协调发展理念",
        "数字经济与双碳战略的辩证关系"
    ]

    print("\n分词测试：")
    for text in test_cases:
        seg = jieba.lcut(text)
        print(f"原文：{text}")
        print(f"分词：{seg}\n")


if __name__ == "__main__":
    # 执行主流程
    processed_df = main()

    # 调试信息输出
    print("\n" + "=" * 30 + " 调试信息 " + "=" * 30)
    print("生成文件列结构：")
    print(processed_df.dtypes)
    print("\n前100字符示例：")
    print(processed_df['draft_clean'].iloc[0][:100])

    # 运行测试需要时取消注释
    # test_segmentation()