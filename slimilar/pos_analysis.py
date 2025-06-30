# pos_analysis.py
# -*- coding: utf-8 -*-
"""
词性分布分析模块 v1.0
功能：独立分析词性分布，生成雷达图所需数据
输入：
  - head.txt  文本1
  - last.txt  文本2
输出：
  - pos_distribution.csv 词性分布数据
"""

import os
import jieba.posseg as pseg
from collections import defaultdict
import pandas as pd

# ================= 配置区 =================
INPUT_CONFIG = {
    "chugao": r"D:\SASanalysis\SAS_text\sample_doc_v1.txt",    
    "zhonggao": r"D:\SASanalysis\SAS_text\lsample_doc_v2.txt"   
}
OUTPUT_PATH = r"D:\SASanalysis\SAS_text\python_SAS\output_wordnum\pos_distribution.csv"  # 输出路径
POS_MAPPING = {  # 词性标签映射
    'n': '名词',
    'v': '动词',
    'a': '形容词',
    'nz': '专业术语'
}
# ==========================================

def create_dir_if_needed(path):
    """自动创建输出目录"""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"已创建目录：{dir_name}")

def load_text(file_path):
    """安全加载文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except UnicodeDecodeError:
        print(f"错误：文件 {file_path} 编码非UTF-8，请转换编码")
        return None

def analyze_pos(text):
    """分析文本词性分布"""
    if not text:
        return defaultdict(int)
    
    counter = defaultdict(int)
    try:
        words = pseg.cut(text)
        for word, flag in words:
            if flag in POS_MAPPING:
                counter[POS_MAPPING[flag]] += 1
    except Exception as e:
        print(f"分词失败：{str(e)}")
    return counter

def generate_distribution_data():
    """生成分布数据"""
    # 加载文本
    data = {
        'chugao': load_text(INPUT_CONFIG['chugao']),
        'zhonggao': load_text(INPUT_CONFIG['zhonggao'])
    }
    
    # 检查数据完整性
    if not all(data.values()):
        print("终止：输入文件缺失或损坏")
        return False

    # 分析词性
    pos_counts = {
        name: analyze_pos(text)
        for name, text in data.items()
    }

    # 构建DataFrame
    categories = ['名词', '动词', '形容词', '专业术语']
    df = pd.DataFrame({
        'category': categories,
        'chugao': [pos_counts['chugao'].get(cat, 0) for cat in categories],
        'zhonggao': [pos_counts['zhonggao'].get(cat, 0) for cat in categories]
    })
    
    # 保存结果
    try:
        create_dir_if_needed(OUTPUT_PATH)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"数据已保存至：{OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"保存失败：{str(e)}")
        return False

if __name__ == "__main__":
    print("==== 开始词性分析 ====")
    if generate_distribution_data():
        print("==== 分析成功完成 ====")
    else:
        print("==== 分析过程中止 ====")