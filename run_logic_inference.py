#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷运行脚本 - 用于运行 models/logic_inference.py
支持选择数据集、模型和备份策略
"""

# ============================================================================
# 快速配置区域 - 在这里修改常用的运行参数
# ============================================================================

# 数据集配置
DATASET_NAME = 'AR-LSAT'  # 可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT', 'self-refine-1_ProntoQA'
DATASET_SPLIT = 'dev'      # 可选: 'dev', 'test'

# 模型配置（需要与 run_logic_program.py 中使用的模型名称一致）
MODEL_NAME = 'glm-4.6'  # 必须与生成逻辑程序时使用的模型名称一致glm-4-flash-250414

# 备份策略配置
BACKUP_STRATEGY = 'random'  # 可选: 'random', 'LLM'
BACKUP_LLM_RESULT_PATH = '../baselines/results'  # 当 backup_strategy='LLM' 时使用

# 其他配置（一般不需要修改）
LOGIC_PROGRAMS_PATH = './outputs/logic_programs'  # 逻辑程序文件路径
SAVE_PATH = './outputs/logic_inference'  # 推理结果保存路径
TIMEOUT = 60  # 程序执行超时时间（秒）

# ============================================================================
# 以下为代码实现部分，一般不需要修改
# ============================================================================

import os
import sys
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 添加 models 目录到路径，以便 logic_inference.py 中的 symbolic_solvers 导入能正常工作
models_dir = os.path.join(project_root, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# 导入 logic_inference 模块
try:
    from models.logic_inference import LogicInferenceEngine
except ImportError as e:
    print(f"错误: 无法导入必要模块: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 支持的数据集
SUPPORTED_DATASETS = ['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT']

# 支持的备份策略
SUPPORTED_BACKUP_STRATEGIES = ['random', 'LLM']


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='便捷运行脚本 - 用于运行 models/logic_inference.py\n\n'
                    '提示: 可以直接在脚本文件开头修改配置（DATASET_NAME, MODEL_NAME等），'
                    '也可以使用命令行参数覆盖这些配置。\n\n'
                    '注意: 此脚本需要先运行 run_logic_program.py 生成逻辑程序文件。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 方式1: 直接运行（使用文件开头的默认配置）
  python run_logic_inference.py

  # 方式2: 使用命令行参数覆盖配置
  python run_logic_inference.py --dataset_name ProntoQA --model_name glm-4 --backup_strategy random

  # 方式3: 只覆盖部分参数
  python run_logic_inference.py --dataset_name FOLIO --split test --backup_strategy LLM

  # 使用 LLM 备份策略（需要提供备份结果路径）
  python run_logic_inference.py --backup_strategy LLM --backup_LLM_result_path ../baselines/results

支持的数据集: {', '.join(SUPPORTED_DATASETS)}
支持的备份策略: {', '.join(SUPPORTED_BACKUP_STRATEGIES)}
        """
    )
    
    # 必需参数（使用文件开头的默认值）
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=DATASET_NAME,
        choices=SUPPORTED_DATASETS,
        help=f'数据集名称 (默认: {DATASET_NAME})'
    )
    
    # 可选参数（使用文件开头的默认值）
    parser.add_argument(
        '--split',
        type=str,
        default=DATASET_SPLIT,
        choices=['dev', 'test'],
        help=f'数据集分割 (默认: {DATASET_SPLIT})'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=MODEL_NAME,
        help=f'模型名称 (默认: {MODEL_NAME}, 必须与生成逻辑程序时使用的模型名称一致)'
    )
    
    parser.add_argument(
        '--logic_programs_path',
        type=str,
        default=LOGIC_PROGRAMS_PATH,
        help=f'逻辑程序文件路径 (默认: {LOGIC_PROGRAMS_PATH})'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default=SAVE_PATH,
        help=f'保存路径 (默认: {SAVE_PATH})'
    )
    
    parser.add_argument(
        '--backup_strategy',
        type=str,
        default=BACKUP_STRATEGY,
        choices=SUPPORTED_BACKUP_STRATEGIES,
        help=f'备份策略 (默认: {BACKUP_STRATEGY})'
    )
    
    parser.add_argument(
        '--backup_LLM_result_path',
        type=str,
        default=BACKUP_LLM_RESULT_PATH,
        help=f'备份LLM结果路径 (默认: {BACKUP_LLM_RESULT_PATH}, 当 backup_strategy=LLM 时使用)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=TIMEOUT,
        help=f'程序执行超时时间（秒） (默认: {TIMEOUT})'
    )
    
    args = parser.parse_args()
    
    return args


def main():
    """主函数"""
    print("=" * 60)
    print("Logic Inference Engine - 便捷运行脚本")
    print("=" * 60)
    print()
    
    # 解析参数
    args = parse_args()
    
    # 显示配置信息
    print("\n配置信息:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  分割: {args.split}")
    print(f"  模型名称: {args.model_name}")
    print(f"  逻辑程序路径: {args.logic_programs_path}")
    print(f"  保存路径: {args.save_path}")
    print(f"  备份策略: {args.backup_strategy}")
    if args.backup_strategy == 'LLM':
        print(f"  备份LLM结果路径: {args.backup_LLM_result_path}")
    print(f"  超时时间: {args.timeout} 秒")
    print()
    
    # 检查逻辑程序文件是否存在
    # 注意: logic_inference.py 中硬编码了路径 './outputs/logic_programs'
    # 所以我们需要检查该路径下的文件
    expected_file = os.path.join('./outputs/logic_programs', 
                                 f'{args.dataset_name}_{args.split}_{args.model_name}.json')
    logic_program_file = os.path.join(
        args.logic_programs_path,
        f'{args.dataset_name}_{args.split}_{args.model_name}.json'
    )
    
    # 如果期望的文件不存在，但用户指定的路径存在，则创建链接或复制
    if not os.path.exists(expected_file):
        if os.path.exists(logic_program_file):
            print(f"提示: logic_inference.py 期望文件在: {expected_file}")
            print(f"但实际文件在: {logic_program_file}")
            print(f"正在创建符号链接或复制文件...")
            # 确保目标目录存在
            os.makedirs('./outputs/logic_programs', exist_ok=True)
            # 如果目标文件不存在，创建符号链接
            if not os.path.exists(expected_file):
                try:
                    os.symlink(os.path.abspath(logic_program_file), 
                              os.path.abspath(expected_file))
                    print(f"已创建符号链接: {expected_file} -> {logic_program_file}")
                except OSError:
                    # 如果符号链接失败（Windows 或权限问题），尝试复制
                    import shutil
                    shutil.copy2(logic_program_file, expected_file)
                    print(f"已复制文件: {logic_program_file} -> {expected_file}")
        else:
            print(f"错误: 逻辑程序文件不存在: {expected_file}")
            if os.path.abspath(logic_program_file) != os.path.abspath(expected_file):
                print(f"也检查了用户指定的路径: {logic_program_file}")
            print(f"\n提示: 请先运行 run_logic_program.py 生成逻辑程序文件。")
            print(f"运行命令示例:")
            print(f"  python run_logic_program.py --dataset_name {args.dataset_name} --split {args.split} --model_name {args.model_name}")
            sys.exit(1)
    elif not os.path.exists(logic_program_file):
        # 如果期望的文件存在，但用户指定的路径不存在，提示用户
        print(f"提示: 找到逻辑程序文件: {expected_file}")
        print(f"（这是 logic_inference.py 期望的默认路径）")
    
    # 如果使用 LLM 备份策略，检查备份结果文件是否存在
    if args.backup_strategy == 'LLM':
        if not os.path.exists(args.backup_LLM_result_path):
            print(f"警告: 备份LLM结果文件不存在: {args.backup_LLM_result_path}")
            print(f"如果文件路径不正确，请使用 --backup_LLM_result_path 参数指定正确的路径。")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # 创建保存目录
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"已创建保存目录: {args.save_path}")
    
    print("=" * 60)
    print("开始执行逻辑推理...")
    print("=" * 60)
    print()
    
    try:
        # 创建推理引擎
        # 注意: LogicInferenceEngine 会从 './outputs/logic_programs' 加载逻辑程序
        # 我们已经确保文件在该路径下存在
        
        # 创建推理引擎
        inference_engine = LogicInferenceEngine(args)
        
        # 运行推理
        inference_engine.inference_on_dataset()
        
        print()
        print("=" * 60)
        print("推理完成!")
        print("=" * 60)
        output_file = os.path.join(
            args.save_path,
            f'{args.dataset_name}_{args.split}_{args.model_name}_backup-{args.backup_strategy}.json'
        )
        print(f"结果已保存到: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断了程序执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

