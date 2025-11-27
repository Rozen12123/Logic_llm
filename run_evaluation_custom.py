#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义评估脚本
在 run_evaluation.py 的基础上，支持直接指定任意推理结果文件。

默认用于统计:
  ./outputs/logic_inference/self-refine-1_ProntoQA_dev_glm-4.5_backup-random.json
"""

# ============================================================================
# 快速配置区域 - 默认参数，可通过命令行覆盖
# ============================================================================

# 默认结果文件（可以直接改成你想评估的文件）
DEFAULT_RESULT_FILE = './outputs/logic_inference/self-refine-1_ProntoQA_dev_glm-4-flash-250414_backup-random.json'

# ============================================================================
# 以下为代码实现部分，一般不需要修改
# ============================================================================

import os
import sys
import argparse

# 添加项目根目录到路径，便于 import models.*
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入 evaluation 模块
try:
    from models.evaluation import full_evaluation
except ImportError as e:
    print(f"错误: 无法导入 models.evaluation: {e}")
    print("请确认依赖已安装: pip install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='自定义评估脚本 - 直接指定推理结果 JSON 文件进行统计\n'
                    '提示: 可直接修改文件顶部 DEFAULT_RESULT_FILE，也可通过命令行参数覆盖。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认的 self-refine 结果文件
  python run_evaluation_custom.py

  # 指定任意结果文件
  python run_evaluation_custom.py --result_file ./outputs/logic_inference/ProntoQA_dev_glm-4.5_backup-random.json
        """
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default=DEFAULT_RESULT_FILE,
        help=f'待评估的推理结果 JSON 文件路径 (默认: {DEFAULT_RESULT_FILE})'
    )

    return parser.parse_args()


def main():
    print("=" * 60)
    print("Custom Evaluation Runner - 自定义统计脚本")
    print("=" * 60)

    args = parse_args()

    result_file = os.path.abspath(args.result_file)

    print("\n配置信息:")
    print(f"  结果文件: {result_file}")
    print()

    if not os.path.exists(result_file):
        print(f"错误: 找不到推理结果文件: {result_file}")
        print("请确认是否已运行相应的推理脚本并生成该输出。")
        sys.exit(1)

    print("=" * 60)
    print("开始评估...")
    print("=" * 60)
    print()

    try:
        full_evaluation(result_file)
        print()
        print("=" * 60)
        print("评估完成!")
        print("=" * 60)
    except Exception as exc:
        print(f"\n错误: 评估过程中出现异常 - {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


