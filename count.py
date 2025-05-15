import os
import statistics
import json

# 指定要跳过的目录或文件（可以添加更多路径）
SKIP_PATHS = {
    'venv',  # 跳过虚拟环境目录
    '.git',  # 跳过Git目录
    '__pycache__',  # 跳过Python缓存目录
    'PaddleOCR-main',
    'ultralytics-8.3.91',
    '.venv',
    'RCAN-master',
    "code"
}

# 指定C/C++代码所在的两个路径
CPP1_PATH = 'D:/vs_c_project/vs_c_project'
CPP2_PATH = 'D:/vscode_c_project'

# 指定C/C++文件的扩展名
CPP_EXTENSIONS = {'.c', '.cpp', '.h', '.hpp'}


def should_skip(path):
    """检查路径是否应该被跳过"""
    for skip_path in SKIP_PATHS:
        if skip_path in path.split(os.sep):
            return True
    return False


def is_cpp_file(file_path):
    """检查是否是C/C++文件"""
    return any(file_path.endswith(ext) for ext in CPP_EXTENSIONS)


def count_code_lines(python_directory='.', cpp_directories=None):
    """
    统计指定目录中所有Python（包括 .py 和 .ipynb 文件）和C/C++文件的有效代码行数（不包含空行），
    并逐步输出当前正在统计的目录及其代码行数

    Args:
        python_directory: 要统计的Python目录，默认为当前目录
        cpp_directories: 要统计的C/C++目录列表

    Returns:
        Python代码行数，C/C++代码行数，Python文件信息列表，C/C++文件信息列表
    """
    if cpp_directories is None:
        cpp_directories = []

    python_total_lines = 0
    cpp_total_lines = 0
    python_files_info = []  # 用于存储每个Python文件的路径和行数
    cpp_files_info = []     # 用于存储每个C/C++文件的路径和行数

    # 统计Python代码行数，包括 .py 和 .ipynb 文件
    print("开始统计Python代码行数...")
    for root, dirs, files in os.walk(python_directory):
        # 跳过不需要统计的目录
        dirs[:] = [d for d in dirs if not should_skip(os.path.join(root, d))]

        current_dir_lines = 0
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_line_count = 0
                        for line in f:
                            stripped_line = line.strip()
                            if stripped_line:
                                file_line_count += 1
                                python_total_lines += 1
                        python_files_info.append({'path': file_path, 'lines': file_line_count})
                        current_dir_lines += file_line_count
                except Exception as e:
                    print(f"警告：无法处理文件 {file_path}：{e}")

            elif file.endswith('.ipynb'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        notebook = json.load(f)
                        file_line_count = 0
                        for cell in notebook['cells']:
                            if cell['cell_type'] == 'code':
                                for line in cell['source']:
                                    stripped_line = line.strip()
                                    if stripped_line:
                                        file_line_count += 1
                                        python_total_lines += 1
                        python_files_info.append({'path': file_path, 'lines': file_line_count})
                        current_dir_lines += file_line_count
                except Exception as e:
                    print(f"警告：无法处理文件 {file_path}：{e}")

        if current_dir_lines > 0:
            print(f"正在统计目录 {root} 的Python代码行数：{current_dir_lines} 行")

    # 统计C/C++代码行数
    print("\n开始统计C/C++代码行数...")
    for cpp_dir in cpp_directories:
        for root, dirs, files in os.walk(cpp_dir):
            current_dir_lines = 0
            for file in files:
                file_path = os.path.join(root, file)
                if is_cpp_file(file_path):
                    try:
                        # 尝试用不同的编码方式打开文件
                        encodings = ['utf-8', 'gbk', 'latin-1']
                        file_line_count = 0
                        for encoding in encodings:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    for line in f:
                                        stripped_line = line.strip()
                                        if stripped_line:
                                            file_line_count += 1
                                            cpp_total_lines += 1
                                    cpp_files_info.append({'path': file_path, 'lines': file_line_count})
                                    current_dir_lines += file_line_count
                                break  # 如果成功读取，就停止尝试其他编码
                            except UnicodeDecodeError:
                                continue  # 尝试下一个编码
                    except Exception as e:
                        print(f"警告：无法处理文件 {file_path}：{e}")

            if current_dir_lines > 0:
                print(f"正在统计目录 {root} 的C/C++代码行数：{current_dir_lines} 行")

    return python_total_lines, cpp_total_lines, python_files_info, cpp_files_info


def analyze_code_statistics(python_files_info, cpp_files_info):
    """分析代码统计信息"""
    analysis = {}

    # Python代码分析
    if python_files_info:
        python_lines = [file['lines'] for file in python_files_info]
        analysis['python_max_file'] = max(python_lines)
        analysis['python_min_file'] = min(python_lines)
        analysis['python_avg_file'] = statistics.mean(python_lines)
        analysis['python_median_file'] = statistics.median(python_lines)
        analysis['python_file_count'] = len(python_files_info)
        analysis['python_max_file_path'] = [file['path'] for file in python_files_info if file['lines'] == analysis['python_max_file']]
        analysis['python_min_file_path'] = [file['path'] for file in python_files_info if file['lines'] == analysis['python_min_file']]
    else:
        analysis['python_max_file'] = 0
        analysis['python_min_file'] = 0
        analysis['python_avg_file'] = 0
        analysis['python_median_file'] = 0
        analysis['python_file_count'] = 0
        analysis['python_max_file_path'] = []
        analysis['python_min_file_path'] = []

    # C/C++代码分析
    if cpp_files_info:
        cpp_lines = [file['lines'] for file in cpp_files_info]
        analysis['cpp_max_file'] = max(cpp_lines)
        analysis['cpp_min_file'] = min(cpp_lines)
        analysis['cpp_avg_file'] = statistics.mean(cpp_lines)
        analysis['cpp_median_file'] = statistics.median(cpp_lines)
        analysis['cpp_file_count'] = len(cpp_files_info)
        analysis['cpp_max_file_path'] = [file['path'] for file in cpp_files_info if file['lines'] == analysis['cpp_max_file']]
        analysis['cpp_min_file_path'] = [file['path'] for file in cpp_files_info if file['lines'] == analysis['cpp_min_file']]
    else:
        analysis['cpp_max_file'] = 0
        analysis['cpp_min_file'] = 0
        analysis['cpp_avg_file'] = 0
        analysis['cpp_median_file'] = 0
        analysis['cpp_file_count'] = 0
        analysis['cpp_max_file_path'] = []
        analysis['cpp_min_file_path'] = []

    return analysis


if __name__ == "__main__":
    # 指定Python目录和C/C++目录列表
    python_dir = '.'  # Python代码所在的目录，默认为当前目录
    cpp_dirs = [CPP1_PATH, CPP2_PATH]  # C/C++代码所在的两个路径

    python_total, cpp_total, python_files_info, cpp_files_info = count_code_lines(python_dir, cpp_dirs)
    analysis = analyze_code_statistics(python_files_info, cpp_files_info)

    print(f"\n项目中所有Python脚本（包括 .ipynb 文件）的总代码行数（不包含空行）：{python_total} 行")
    print(f"指定路径下所有C/C++脚本的总代码行数（不包含空行）：{cpp_total} 行")
    print(f"项目中所有代码的总行数（不包含空行）：{python_total + cpp_total} 行")

    # 输出统计分析
    print("\n=== 统计分析 ===")
    print(f"Python文件数量（包括 .ipynb 文件）：{analysis['python_file_count']} 个")
    print(f"C/C++文件数量：{analysis['cpp_file_count']} 个")
    print(f"所有文件数量：{analysis['python_file_count'] + analysis['cpp_file_count']} 个")

    # Python文件分析
    if analysis['python_file_count'] > 0:
        print(f"\nPython文件最大代码行数：{analysis['python_max_file']}（文件：{', '.join(analysis['python_max_file_path'])}）")
        print(f"Python文件最小代码行数：{analysis['python_min_file']}（文件：{', '.join(analysis['python_min_file_path'])}）")
        print(f"Python文件平均代码行数：{analysis['python_avg_file']:.2f}")
        print(f"Python文件中位数代码行数：{analysis['python_median_file']}")
    else:
        print("\n没有找到Python文件")

    # C/C++文件分析
    if analysis['cpp_file_count'] > 0:
        print(f"\nC/C++文件最大代码行数：{analysis['cpp_max_file']}（文件：{', '.join(analysis['cpp_max_file_path'])}）")
        print(f"C/C++文件最小代码行数：{analysis['cpp_min_file']}（文件：{', '.join(analysis['cpp_min_file_path'])}）")
        print(f"C/C++文件平均代码行数：{analysis['cpp_avg_file']:.2f}")
        print(f"C/C++文件中位数代码行数：{analysis['cpp_median_file']}")
    else:
        print("\n没有找到C/C++文件")