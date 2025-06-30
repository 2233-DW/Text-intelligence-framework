# -*- coding: utf-8 -*-
"""
主题词典生成器 v2.0
功能：按比例生成可定制规模的混合词典
"""
import os
import fitz
import docx
import jieba
import jieba.analyse
from collections import defaultdict
from typing import Dict, Set

# ================= 配置区 =================
PAPER_DIR = r"D:\SASanalysis\SAS_text\dictionary_create"
THEME_DICT_PATH = r"D:\SASanalysis\SAS_text\sample_doc_v1.txt"
OUTPUT_DICT = r"D:\SASanalysis\SAS_text\combined_dict.txt"
COMMON_WORDS_PATH = r"D:\SASanalysis\SAS_text\common_words.txt"

DICT_SETTINGS = {
    'total_words': 100,
    'cross_ratio': 0.2,
    'paper_ratio': 0.5,
    'theme_ratio': 0.5
}

EXTRACT_SETTINGS = {
    'topK': 200,
    'withWeight': False,
    'allowPOS': ('n', 'vn', 'ns'),
    'stop_words': None
}


# =========================================

def safe_read_file(file_path: str) -> str:
    """安全读取不同格式文件"""
    text = ""
    try:
        if file_path.endswith('.pdf'):
            with fitz.open(file_path) as doc:
                text = "".join([page.get_text() for page in doc])
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"文件读取失败: {os.path.basename(file_path)} - {str(e)}")
        return ""


def extract_paper_keywords() -> Dict[str, int]:
    """从文中提取候选关键词"""
    word_freq = defaultdict(int)

    valid_files = 0
    for filename in os.listdir(PAPER_DIR):
        file_path = os.path.join(PAPER_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        text = safe_read_file(file_path)
        if len(text) < 100:
            continue

        try:
            words = jieba.analyse.extract_tags(
                text,
                topK=EXTRACT_SETTINGS['topK'],
                allowPOS=EXTRACT_SETTINGS['allowPOS']
            )
            for word in words:
                word_freq[word] += 1
            valid_files += 1
        except Exception as e:
            print(f"关键词提取失败: {filename} - {str(e)}")

    if valid_files == 0:
        print("警告：未发现有效文件")
    return word_freq


def load_existing_dict() -> Set[str]:
    """加载现有船山术语词典"""
    try:
        with open(THEME_DICT_PATH, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(f"词典文件不存在：{THEME_DICT_PATH}")
        return set()
    except Exception as e:
        print(f"加载词典失败: {str(e)}")
        return set()


def generate_combined_dict(paper_words: Dict[str, int], theme_words: Set[str]) -> dict:
    """按比例生成词典并返回统计信息"""
    # 参数校验
    if not isinstance(paper_words, dict) or not isinstance(theme_words, set):
        print("无效的输入数据类型")
        return {'combined': [], 'stats': {}}

    # 计算各区间目标词数
    total = DICT_SETTINGS['total_words']
    cross_target = int(total * DICT_SETTINGS['cross_ratio'])
    remain = total - cross_target

    # 获取候选词
    cross_words = sorted(
        [w for w in paper_words if w in theme_words],
        key=lambda x: paper_words[x],
        reverse=True
    )

    # 保存交叉词
    with open(COMMON_WORDS_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(cross_words))
        print(f"交叉词汇已保存至: {COMMON_WORDS_PATH}")

    # 动态分配词数
    cross_actual = min(cross_target, len(cross_words))
    paper_target = int(remain * DICT_SETTINGS['paper_ratio'])
    paper_candidates = [w for w in paper_words if w not in theme_words]
    paper_candidates = sorted(paper_candidates, key=lambda x: paper_words[x], reverse=True)
    paper_actual = min(paper_target, len(paper_candidates))

    theme_target = remain - paper_actual
    theme_candidates = [w for w in theme_words if w not in paper_words]
    theme_actual = min(theme_target, len(theme_candidates))

    # 组合结果
    combined = (
            cross_words[:cross_actual] +
            paper_candidates[:paper_actual] +
            theme_candidates[:theme_actual]
    )

    # 缺口处理
    gap = total - len(combined)
    backup = []
    if gap > 0:
        # 优先补充论文词
        remaining_paper = paper_candidates[paper_actual:paper_actual + gap]
        backup += remaining_paper
        gap -= len(remaining_paper)

        # 再补充词语
        if gap > 0:
            remaining_theme = theme_candidates[theme_actual:theme_actual + gap]
            backup += remaining_theme

    combined += backup[:total - len(combined)]

    return {
        'combined': combined[:total],
        'stats': {
            'cross': cross_actual,
            'paper': len(paper_candidates[:paper_actual]),
            'theme': len(theme_candidates[:theme_actual]),
            'added': len(backup[:total - len(combined)])
        }
    }

def main():
    print("开始生成主题词典...")

    # 阶段1：提取论文关键词
    print("正在分析...")
    paper_words = extract_paper_keywords()
    if not paper_words:
            print("关键词提取失败，请检查输入文件")
            return

        # 阶段2：加载现有词典
    print("加载船山术语...")
    theme_words = load_existing_dict()
    if not theme_words:
        print("词典加载失败，请检查词典文件")
        return

        # 阶段3：生成组合词典
    print("组合词典内容...")
    result = generate_combined_dict(paper_words, theme_words)

    # 保存结果
    with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
        f.write("\n".join(result['combined']))

        # 统计输出
        stats = result['stats']
        print("\n 词典构成统计:")
        print("=" * 40)
        print(f"总词数: {len(result['combined'])}")
        print(f"交叉词汇: {stats['cross']} ({stats['cross'] / len(result['combined']):.1%})")
        print(f"论文特有: {stats['paper']} ({stats['paper'] / len(result['combined']):.1%})")
        print(f"术语特有: {stats['theme']} ({stats['theme'] / len(result['combined']):.1%})")
        if stats['added'] > 0:
            print(f"缺口补足: {stats['added']} 词")
        print(f"保存路径: {OUTPUT_DICT}")

if __name__ == "__main__":
    # 初始化结巴分词
     if EXTRACT_SETTINGS['stop_words']:
        jieba.analyse.set_stop_words(EXTRACT_SETTINGS['stop_words'])

    # 执行主程序
     main()
