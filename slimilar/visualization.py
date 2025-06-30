# -*- coding: utf-8 -*-
"""
可视化分析脚本 v3.3
修复内容：
1. 补全缺失的 export_diff_words 函数
2. 统一异常处理逻辑
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import os

# ================= 配置区 =================
MATRIX_PATH = r"D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1\tfidf_matrix_2.csv"
SIM_MATRIX_PATH = r"D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\cos_sim_result4.csv"
POS_DATA_PATH = r"D:\SASanalysis\SAS_text\python_SAS\output_wordnum\pos_distribution.csv"

OUTPUT_DIR = r"D:\SASanalysis\SAS_text\python_SAS\output_keshihua"
TOP_N = 30
DOC_NAMES = ["初稿", "终稿"]

output_config = {
    'static_diff': os.path.join(OUTPUT_DIR, 'feature_diff.png'),
    'interactive_diff': os.path.join(OUTPUT_DIR, 'feature_diff.html'),
    'heatmap': os.path.join(OUTPUT_DIR, 'heatmap.png'),
    'radar': os.path.join(OUTPUT_DIR, 'radar_compare.png'),
    'diff_csv': os.path.join(OUTPUT_DIR, 'top_diff_words.csv')
}
# ==========================================

def load_data(path, is_matrix=True):
    """增强版数据加载"""
    try:
        df = pd.read_csv(path, index_col=0) if is_matrix else pd.read_csv(path)
        print(f"✅ 成功加载数据：{os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"❌ 加载失败：{os.path.basename(path)} - {str(e)}")
        raise

# ----------------- 核心功能模块 -----------------
def plot_feature_diff(df, top_n=30):
    """静态差异图"""
    plt.rcParams.update({'font.sans-serif': 'SimHei', 'axes.unicode_minus': False})

    diff = df.diff().abs().iloc[1]
    top_diff = diff.nlargest(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=top_diff.values,
        y=top_diff.index,
        hue=top_diff.index,  # 关键修复
        palette="viridis",
        legend=False,
        dodge=False
    )
    plt.title(f"TOP {top_n} 差异特征词")
    plt.savefig(output_config['static_diff'], dpi=300)
    print(f"📊 静态图保存至：{output_config['static_diff']}")

def interactive_plot(df, top_n=30):
    """交互式可视化"""
    try:
        diff = df.diff().abs().iloc[1]
        top_diff = diff.nlargest(top_n)

        fig = px.bar(
            top_diff,
            orientation='h',
            labels={'index': '特征词', 'value': '差异值'},
            title=f"TOP {top_n} 差异特征词",
            color=top_diff.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            hovermode='closest',
            font=dict(family='SimHei')  # 解决中文显示问题
        )
        fig.write_html(output_config['interactive_diff'])
        print(f"🖱 交互图已保存：{output_config['interactive_diff']}")
    except Exception as e:
        print(f"❌ 交互图生成失败：{str(e)}")

def export_diff_words(df, top_n=30):  # 补全缺失函数
    """导出差异词数据"""
    try:
        diff = df.diff().abs().iloc[1]
        diff.nlargest(top_n).to_csv(
            output_config['diff_csv'],
            header=['差异值'],
            encoding='utf-8-sig'  # 确保中文兼容
        )
        print(f"💾 差异数据已导出：{output_config['diff_csv']}")
    except Exception as e:
        print(f"❌ 数据导出失败：{str(e)}")

def plot_similarity_heatmap():
    """相似度热力图"""
    # 加载或计算相似度矩阵
    if os.path.exists(SIM_MATRIX_PATH):
        sim_matrix = load_data(SIM_MATRIX_PATH).values
    else:
        tfidf_matrix = load_data(MATRIX_PATH).values
        sim_matrix = cosine_similarity(tfidf_matrix)
        print("⚠️ 注意：使用实时计算的余弦相似度矩阵")

    # 绘图设置
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=DOC_NAMES,
        yticklabels=DOC_NAMES
    )
    plt.title("文档相似度热力图")
    plt.savefig(output_config['heatmap'], dpi=300, bbox_inches='tight')
    print(f"🌡 热力图已保存：{output_config['heatmap']}")

def plot_pos_radar():
    """词性雷达图"""
    try:
        pos_data = load_data(POS_DATA_PATH, is_matrix=False)

        # 数据校验
        if not {'category', 'chugao', 'zhonggao'}.issubset(pos_data.columns):
            raise ValueError("CSV 文件缺少必要列")

        categories = pos_data['category'].tolist()
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, pos_data['chugao'], 'b-', label='初稿')
        ax.fill(angles, pos_data['chugao'], 'b', alpha=0.1)
        ax.plot(angles, pos_data['zhonggao'], 'r-', label='终稿')
        ax.fill(angles, pos_data['zhonggao'], 'r', alpha=0.1)
        ax.set_thetagrids(np.degrees(angles), categories)
        plt.legend()
        plt.savefig(output_config['radar'], dpi=300)
        print(f"📉 雷达图保存至：{output_config['radar']}")
    except Exception as e:
        print(f"⚠️ 跳过雷达图：{str(e)}")

# ----------------- 主流程控制 -----------------
if __name__ == "__main__":
    print("==== 可视化分析开始 ====")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        tfidf_df = load_data(MATRIX_PATH)
        plot_feature_diff(tfidf_df, TOP_N)
        interactive_plot(tfidf_df, TOP_N)
        export_diff_words(tfidf_df, TOP_N)  # 现在可正常调用
        plot_similarity_heatmap()
        plot_pos_radar()
    except Exception as e:
        print(f"🛑 主流程异常：{str(e)}")
        exit(1)

    print("\n==== 分析完成 ====")
    [print(f"- {os.path.basename(v)}") for v in output_config.values()]