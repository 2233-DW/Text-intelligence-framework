# -*- coding: utf-8 -*-
"""
自动化流水线监控服务 v1.1
更新：增加输出文件完整性检查
"""

import os
import time
import jieba
import subprocess
import pandas as pd  # 新增必要库导入
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= 配置区 =================
WATCH_DIRS = [
    r"D:\SASanalysis\SAS_text",
    r"D:\SASanalysis\SAS_text\python_SAS"
]
RELOAD_DELAY = 5
SCRIPTS_ORDER = [
    r"D:\SASanalysis\SAS_text\python_SAS\preprocess.py",
    r"D:\SASanalysis\SAS_text\tfidf_analysis3.py",
    r"D:\SASanalysis\SAS_text\python_SAS\visualization.py",
    r"D:\SASanalysis\SAS_run\cos_sim_3.sas"
]

OUTPUT_FILES = [  # 新增输出文件配置
    r"D:\SASanalysis\SAS_text\python_SAS\output_yuchuli\text_pairs_2.csv",
    r"D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1\tfidf_matrix_2.csv",
    r"D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\cos_sim_result4.csv"
]
OUTPUT_DICT = r"D:\SASanalysis\SAS_text\combined_dict.txt"  # 新增配置项
# ================= 配置区 =================
# ...（原有配置不变）

PREVIEW_SETTINGS = {  # 新增预览配置
    'preview_lines': 3,          # 每个文件预览行数
    'max_columns': 5,            # 最大显示列数
    'sample_text_length': 100,   # 文本采样长度
    'target_files': {            # 指定需要预览的文件及其方式
        r"D:\SASanalysis\SAS_text\python_SAS\output_yuchuli\text_pairs_2.csv": 'text',
        r"D:\SASanalysis\SAS_text\python_SAS\output_jianmo_1\tfidf_matrix_2.csv": 'matrix',
        r"D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\cos_sim_result4.csv": 'similarity'
    }
}
# =========================================
# =========================================
def preview_file_content(file_path, preview_type):
    """安全预览文件内容"""
    try:
        if not os.path.exists(file_path):
            return f"⚠️ 文件不存在: {os.path.basename(file_path)}"

        # 统一使用pandas读取保障编码兼容性
        df = pd.read_csv(file_path, encoding='utf_8_sig', nrows=PREVIEW_SETTINGS['preview_lines'])

        # 根据文件类型生成预览
        if preview_type == 'text':
            sample_text = df.iloc[0]['draft_clean'][:PREVIEW_SETTINGS['sample_text_length']]
            return f"文本样例: {sample_text}..."
        elif preview_type == 'matrix':
            # 限制显示列数
            cols = df.columns[:PREVIEW_SETTINGS['max_columns']]
            return df[cols].head().to_string(index=False)
        elif preview_type == 'similarity':
            return df.head(PREVIEW_SETTINGS['preview_lines']).to_string(index=False)
        else:
            return "未知预览类型"
    except Exception as e:
        return f"预览失败: {str(e)}"


# ============================================
def check_output():
    """文件完整性检查（版本1.1新增）"""
    missing = [f for f in OUTPUT_FILES if not os.path.exists(f)]
    if missing:
        print(f"⚠️ 警告：缺失 {len(missing)} 个关键输出文件")
        for i, f in enumerate(missing, 1):
            print(f"{i}. {os.path.basename(f)} @ {os.path.dirname(f)}")
    else:
        print("✅ 所有输出文件验证通过")
        print("\n当前输出文件状态：")
        for f in OUTPUT_FILES:
            mtime = time.ctime(os.path.getmtime(f))
            print(f"• {os.path.basename(f)} | 最后更新：{mtime}")


class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_trigger = 0

    def on_modified(self, event):
        if time.time() - self.last_trigger < RELOAD_DELAY:
            return

        if any(ext in event.src_path for ext in ['.txt', '.py', '.sas']):
            print(f"\n🔍 检测到变更：{os.path.basename(event.src_path)}")
            self.run_pipeline()
            self.last_trigger = time.time()
            check_output()  # 流程结束后检查

    def run_pipeline(self):
        """执行全流程（版本1.1优化）"""
        try:
            # 前置检查
            if not all(os.path.exists(f) for f in OUTPUT_FILES[:1]):
                print("⏳ 正在初始化缺失的输入文件...")
                subprocess.run(['python', SCRIPTS_ORDER[0]], check=True)

            # 执行流程
            for idx, script in enumerate(SCRIPTS_ORDER):
                print(f"\n🚀 步骤 {idx + 1}/{len(SCRIPTS_ORDER)}: 执行 {os.path.basename(script)}")

                if script.endswith('.py'):
                    self.run_python(script)
                elif script.endswith('.sas'):
                    self.run_sas(script)

                if idx in [0, 1, 3]:  # 阶段检查
                    check_output()

            print("\n🎉 全流程更新完成！")
            self.print_success()

        except subprocess.CalledProcessError as e:
            print(f"❌ 执行失败于 {os.path.basename(script)}")
            print(f"错误详情：\n{e.stderr.decode('gbk')}")
        except Exception as e:
            print(f"🛑 未捕获异常：{str(e)}")

    def run_python(self, script):
        """执行Python脚本"""
        result = subprocess.run(
            ['python', script],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        self.handle_result(result, script)

    def run_sas(self, script):
        """执行SAS脚本（修复版）"""
        sas_exe = r'D:\SASHome\SASFoundation\9.4\sas.exe'
        log_file = r"D:\SASanalysis\SAS_text\python_SAS\out_sasjisuan_1\sas_execution.log"

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # 构建 SAS 命令
        command = f'"{sas_exe}" -sysin "{script}" -log "{log_file}" -nologo'
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='gbk'  # SAS 日志通常使用系统本地编码（如 GBK）
        )

        # 错误处理
        if result.returncode != 0:
            self.handle_failure(script, log_file)
        else:
            self.handle_success(script)

    def handle_result(self, result, script):
        """统一处理执行结果"""
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=script,
                output=result.stdout,
                stderr=result.stderr
            )
        print(f"✔️ {os.path.basename(script)} 执行成功")

    def handle_failure(self, script, log_path):
        """处理 SAS 执行失败（新增方法）"""
        try:
            with open(log_path, 'r', encoding='gbk') as f:
                logs = f.read()
            print(f"❌ SAS 执行失败: {os.path.basename(script)}")
            print("=" * 50)
            print(logs[-1000:])  # 仅输出最后 1000 字符避免刷屏
            print("=" * 50)
        except Exception as e:
            print(f"⚠️ 无法读取日志文件: {str(e)}")

    def handle_success(self, script):
        """处理成功执行"""
        print(f"✔️ {os.path.basename(script)} SAS 执行成功")

    # ================= 新增函数 =================

    # ============================================
    def print_success(self):
        """成功状态打印"""
        print("\n" + "=" * 40)
        print("当前系统状态：")
        check_output()
        # 新增预览模块
        print("\n" + "=" * 40)
        print("关键结果预览：")
        for file_path, preview_type in PREVIEW_SETTINGS['target_files'].items():
            print(f"\n▌ 文件: {os.path.basename(file_path)}")
            print(preview_file_content(file_path, preview_type))
        print("=" * 40 + "\n")


if __name__ == "__main__":
    # 启动时首次检查
    print("🔎 正在初始化文件检查...")
    check_output()

    # 启动监控
    event_handler = ReloadHandler()
    observer = Observer()
    for directory in WATCH_DIRS:
        observer.schedule(event_handler, directory, recursive=True)

    print("\n🖥 监控服务已启动，等待文件变更...")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()