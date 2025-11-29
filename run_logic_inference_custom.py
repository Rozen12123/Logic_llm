#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义逻辑推理脚本
在 run_logic_inference.py 的基础上，支持直接指定任意逻辑程序文件
"""

# ============================================================================
# 快速配置区域 - 默认参数，可通过命令行覆盖
# ============================================================================

# 默认数据集（可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'）
DATASET_NAME = 'ProntoQA'
# 默认数据切分（可选: 'dev', 'test'）
DATASET_SPLIT = 'dev'
# 生成/推理时使用的模型名称，需与逻辑程序文件一致
MODEL_NAME = 'glm-4.5'
# 推理失败备份策略（可选: 'random', 'LLM'）
BACKUP_STRATEGY = 'random'
# 当 BACKUP_STRATEGY='LLM' 时，提供备份答案的文件目录
BACKUP_LLM_RESULT_PATH = '../baselines/results'
# 逻辑程序所在目录（默认 outputs/logic_programs，可自定义）
# 逻辑程序所在目录（默认 outputs/logic_programs，可自定义）
LOGIC_PROGRAMS_PATH = './outputs/logic_programs'
# 默认要加载的逻辑程序文件；设置为 None 则按常规命名查找
DEFAULT_LOGIC_PROGRAM_FILE = './outputs/logic_programs/self-refine-1_ProntoQA_dev_glm-4-flash-250414.json'
# 推理结果保存目录
SAVE_PATH = './outputs/logic_inference'
# 自定义输出文件前缀（例如 'self-refine-1_'）；留空则按默认命名。
# 如果保持为空，脚本会尝试根据 logic_program_file 的文件名前缀自动推断
OUTPUT_PREFIX = ''
# 每个程序执行超时时间，单位秒
TIMEOUT = 60

# ============================================================================
# 以下为实现
# ============================================================================

import argparse
import os
import shutil
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

models_dir = os.path.join(project_root, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

try:
    from models.logic_inference import LogicInferenceEngine
except ImportError as e:
    print(f"错误: 无法导入必要模块: {e}")
    sys.exit(1)

SUPPORTED_DATASETS = ['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT']
SUPPORTED_BACKUP_STRATEGIES = ['random', 'LLM']


def infer_output_prefix_from_logic_file(logic_program_file: str) -> str:
    """
    根据逻辑程序文件名自动推断输出前缀。
    例如:
      self-refine-2_ProntoQA_dev_glm-4-flash-250414.json
    推断出:
      self-refine-2_
    其它情况返回空字符串。
    """
    if not logic_program_file:
        return ''
    name = os.path.basename(logic_program_file)
    if name.startswith('self-refine-'):
        parts = name.split('_', 2)
        if len(parts) >= 2:
            return parts[0] + '_'
    return ''


def parse_args():
    parser = argparse.ArgumentParser(
        description='自定义逻辑推理脚本（支持指定逻辑程序文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python run_logic_inference_custom.py --dataset_name ProntoQA \\
     --model_name glm-4.5 --logic_program_file ./outputs/logic_programs/self-refine-1_ProntoQA_dev_glm-4.5.json

支持的数据集: {', '.join(SUPPORTED_DATASETS)}
支持的备份策略: {', '.join(SUPPORTED_BACKUP_STRATEGIES)}
        """
    )

    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME,
                        choices=SUPPORTED_DATASETS, help='数据集名称')
    parser.add_argument('--split', type=str, default=DATASET_SPLIT,
                        choices=['dev', 'test'], help='数据集分割')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help='模型名称（需与生成时一致）')
    parser.add_argument('--logic_programs_path', type=str, default=LOGIC_PROGRAMS_PATH,
                        help='逻辑程序默认目录')
    parser.add_argument('--logic_program_file', type=str, default=DEFAULT_LOGIC_PROGRAM_FILE,
                        help='待测试的逻辑程序文件路径（可为任意文件）')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='推理结果保存路径')
    parser.add_argument('--backup_strategy', type=str, default=BACKUP_STRATEGY,
                        choices=SUPPORTED_BACKUP_STRATEGIES, help='备份策略')
    parser.add_argument('--backup_LLM_result_path', type=str, default=BACKUP_LLM_RESULT_PATH,
                        help='LLM 备份结果路径（当 backup_strategy=LLM 时使用）')
    parser.add_argument('--timeout', type=int, default=TIMEOUT,
                        help='执行超时时间（秒）')

    return parser.parse_args()


def main():
    print("=" * 60)
    print("Custom Logic Inference Runner")
    print("=" * 60)

    args = parse_args()

    print("\n配置信息:")
    for key, value in [
        ("数据集", args.dataset_name),
        ("分割", args.split),
        ("模型名称", args.model_name),
        ("备份策略", args.backup_strategy),
        ("逻辑程序目录", args.logic_programs_path),
        ("自定义逻辑程序文件", args.logic_program_file or "(使用默认命名文件)"),
        ("保存路径", args.save_path),
    ]:
        print(f"  {key}: {value}")
    print()

    os.makedirs(args.logic_programs_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    expected_file = os.path.join(
        args.logic_programs_path,
        f'{args.dataset_name}_{args.split}_{args.model_name}.json'
    )

    cleanup_needed = False
    backup_file = None

    def restore_original():
        if not cleanup_needed:
            return
        if backup_file and os.path.exists(backup_file):
            shutil.move(backup_file, expected_file)
        elif os.path.exists(expected_file):
            os.remove(expected_file)

    if args.logic_program_file:
        custom_path = os.path.abspath(args.logic_program_file)
        if not os.path.exists(custom_path):
            print(f"错误: 找不到指定的逻辑程序文件: {custom_path}")
            sys.exit(1)

        expected_abs = os.path.abspath(expected_file)

        if os.path.exists(expected_file) and not os.path.samefile(custom_path, expected_file):
            backup_file = expected_file + '.bak'
            shutil.copy2(expected_file, backup_file)

        if not os.path.samefile(custom_path, expected_abs):
            shutil.copy2(custom_path, expected_file)
            cleanup_needed = True

    print("=" * 60)
    print("开始执行逻辑推理...")
    print("=" * 60)
    
    try:
        inference_engine = LogicInferenceEngine(args)
        inference_engine.inference_on_dataset()
        print("\n推理完成!")
        
        # 逻辑推理内部默认的输出文件名（无前缀）
        base_filename = f"{args.dataset_name}_{args.split}_{args.model_name}_backup-{args.backup_strategy}.json"
        base_output_path = os.path.join(args.save_path, base_filename)

        # 根据逻辑程序文件名确定“期望”的结果文件名：
        #   <逻辑程序文件名去掉 .json>_backup-<strategy>.json
        # 例如:
        #   DEFAULT_LOGIC_PROGRAM_FILE = ./outputs/logic_programs/self-refine-2_ProntoQA_dev_glm-4-flash-250414.json
        #   输出: ./outputs/logic_inference/self-refine-2_ProntoQA_dev_glm-4-flash-250414_backup-random.json
        if args.logic_program_file:
            logic_name = os.path.basename(args.logic_program_file)
            stem = logic_name[:-5] if logic_name.endswith('.json') else logic_name
            result_filename = f"{stem}_backup-{args.backup_strategy}.json"
        else:
            # 退回到原来的基于数据集/模型的命名方式
            result_filename = base_filename

        result_path = os.path.join(args.save_path, result_filename)

        # 如果默认输出存在且目标文件名不同，则重命名为期望的结果文件名
        if os.path.exists(base_output_path) and base_output_path != result_path:
            os.rename(base_output_path, result_path)

        print(f"结果已保存到: {result_path}")
    except Exception as exc:
        print(f"\n错误: {exc}")
        import traceback
        traceback.print_exc()
        restore_original()
        sys.exit(1)

    restore_original()


if __name__ == '__main__':
    main()

