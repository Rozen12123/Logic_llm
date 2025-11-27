#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷运行脚本 - 用于运行 models/evaluation.py
对 run_logic_inference.py 的结果文件做统一评测
"""

# ============================================================================
# 快速配置区域 - 在这里修改常用的运行参数
# ============================================================================

# 数据配置
DATASET_NAME = 'ProntoQA'   # 可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'
DATASET_SPLIT = 'dev'       # 可选: 'dev', 'test'

# 模型配置（需与推理阶段一致）
MODEL_NAME = 'glm-4-flash-250414'

# 备份策略（需要与 run_logic_inference.py 的 backup_strategy 对齐）
BACKUP_STRATEGY = 'random'  # 可选: 'random', 'LLM'

# 结果路径
INFERENCE_RESULT_PATH = './outputs/logic_inference'

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

# 支持的数据集
SUPPORTED_DATASETS = ['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT']
# 支持的备份策略
SUPPORTED_BACKUP_STRATEGIES = ['random', 'LLM']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='便捷运行脚本 - 用于运行 models/evaluation.py\n'
                    '提示: 可直接修改文件顶部默认配置，也可通过命令行参数覆盖。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  python run_evaluation.py
  python run_evaluation.py --dataset_name FOLIO --split test --model_name glm-4 --backup_strategy random

支持的数据集: {', '.join(SUPPORTED_DATASETS)}
支持的备份策略: {', '.join(SUPPORTED_BACKUP_STRATEGIES)}
        """
    )

    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME,
                        choices=SUPPORTED_DATASETS, help=f'数据集名称 (默认: {DATASET_NAME})')
    parser.add_argument('--split', type=str, default=DATASET_SPLIT,
                        choices=['dev', 'test'], help=f'数据集分割 (默认: {DATASET_SPLIT})')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f'模型名称 (默认: {MODEL_NAME})')
    parser.add_argument('--backup_strategy', type=str, default=BACKUP_STRATEGY,
                        choices=SUPPORTED_BACKUP_STRATEGIES, help=f'备份策略 (默认: {BACKUP_STRATEGY})')
    parser.add_argument('--result_path', type=str, default=INFERENCE_RESULT_PATH,
                        help=f'推理结果所在目录 (默认: {INFERENCE_RESULT_PATH})')

    return parser.parse_args()


def main():
    print("=" * 60)
    print("Evaluation Runner - 便捷统计脚本")
    print("=" * 60)

    args = parse_args()

    print("\n配置信息:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  分割: {args.split}")
    print(f"  模型: {args.model_name}")
    print(f"  备份策略: {args.backup_strategy}")
    print(f"  结果目录: {args.result_path}")
    print()

    result_file = os.path.join(
        args.result_path,
        f"{args.dataset_name}_{args.split}_{args.model_name}_backup-{args.backup_strategy}.json"
    )

    if not os.path.exists(result_file):
        print(f"错误: 找不到推理结果文件: {result_file}")
        print("请确认是否已运行 run_logic_inference.py 并生成对应输出。")
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

