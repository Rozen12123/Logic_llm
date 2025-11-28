#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷运行脚本 - 用于运行 models/self_refinement.py
支持选择数据集、自我修正轮数以及API相关配置。
"""

# ============================================================================
# 快速配置区域 - 在这里修改常用的运行参数
# ============================================================================

# 数据集配置
DATASET_NAME = 'ProofWriter'      # 可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'
DATASET_SPLIT = 'dev'          # 可选: 'dev', 'test'

# 自我修正轮数
MAXIMUM_ROUNDS = 3             # 自我修正轮数，如 1 / 2 / 3

# 并发设置（预留，与 models/self_refinement.py 中的实现配合使用）
MAX_CONCURRENT = 1             # 当前自我修正内部仍为顺序执行，此参数为预留配置点

# 备份答案策略（与推理阶段保持一致）
BACKUP_STRATEGY = 'random'     # 可选: 'random', 'LLM'
BACKUP_LLM_RESULT_PATH = '../baselines/results'

# API 配置（用于自我修正调用的大模型）
API_PROVIDER = 'zhipuai'       # 可选: 'openai', 'zhipuai'
MODEL_NAME = 'glm-4-flash-250414'  # openai: gpt-4 等; zhipuai: glm-4.5, glm-4-flash-250414 等
MAX_NEW_TOKENS = 10000
STOP_WORDS = '------'
TIMEOUT = 60

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

# 再将 models 目录单独加入 sys.path，保证可以直接导入 symbolic_solvers.*
models_dir = os.path.join(project_root, 'models')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

# 尝试导入 config_loader（用于读取 API Key / provider）
try:
    from config_loader import load_api_key, load_api_provider
except ImportError:
    def load_api_key(provider="zhipuai"):
        return None
    def load_api_provider():
        return "zhipuai"

# 导入 self_refinement 模块
try:
    from models.self_refinement import SelfRefinementEngine, parse_args as base_parse_args
except ImportError as e:
    print(f"错误: 无法导入 models.self_refinement: {e}")
    print("请确认依赖已安装: pip install -r requirements.txt")
    sys.exit(1)


def parse_args():
    """解析命令行参数（在 models/self_refinement.py 的基础上增加默认值）"""
    parser = argparse.ArgumentParser(
        description='便捷运行脚本 - 用于运行 models/self_refinement.py\n'
                    '提示: 可直接修改文件顶部默认配置，也可通过命令行参数覆盖。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用文件顶部的默认配置进行 1 轮自我修正
  python run_self_refinement.py

  # 指定数据集和轮数
  python run_self_refinement.py --dataset_name ProntoQA --split dev --maximum_rounds 3

  # 使用 OpenAI 作为自我修正模型
  python run_self_refinement.py --api_provider openai --model_name gpt-4
        """
    )

    parser.add_argument('--maximum_rounds', type=int, default=MAXIMUM_ROUNDS,
                        help=f'自我修正轮数 (默认: {MAXIMUM_ROUNDS})')
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME,
                        choices=['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'],
                        help=f'数据集名称 (默认: {DATASET_NAME})')
    parser.add_argument('--split', type=str, default=DATASET_SPLIT,
                        choices=['dev', 'test'], help=f'数据集分割 (默认: {DATASET_SPLIT})')
    parser.add_argument('--backup_strategy', type=str, default=BACKUP_STRATEGY,
                        choices=['random', 'LLM'], help=f'备份策略 (默认: {BACKUP_STRATEGY})')
    parser.add_argument('--backup_LLM_result_path', type=str, default=BACKUP_LLM_RESULT_PATH,
                        help=f'LLM 备份结果路径 (默认: {BACKUP_LLM_RESULT_PATH})')
    parser.add_argument('--api_provider', type=str, default=API_PROVIDER,
                        choices=['openai', 'zhipuai'],
                        help=f'API 提供商 (默认: {API_PROVIDER})')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        help=f'自我修正使用的模型名称 (默认: {MODEL_NAME})')
    parser.add_argument('--timeout', type=int, default=TIMEOUT,
                        help=f'执行超时时间(秒) (默认: {TIMEOUT})')
    parser.add_argument('--api_key', type=str,
                        help='API Key (默认: 从配置或环境变量读取)')
    parser.add_argument('--stop_words', type=str, default=STOP_WORDS,
                        help=f'停止词 (默认: {STOP_WORDS})')
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS,
                        help=f'最大新 token 数 (默认: {MAX_NEW_TOKENS})')
    parser.add_argument('--max_concurrent', type=int, default=MAX_CONCURRENT,
                        help=f'自我修正阶段的预留并发数配置点 (当前实现仍为顺序执行)')

    args = parser.parse_args()

    # 如果命令行没有提供 api_provider，则从配置读取
    if not args.api_provider:
        args.api_provider = load_api_provider()
        print(f"使用API提供商: {args.api_provider}")

    # 如果命令行没有提供 api_key，则从配置或环境变量读取
    if not args.api_key:
        args.api_key = load_api_key(args.api_provider)
        if args.api_key:
            print(f"已从配置文件或环境变量读取 {args.api_provider.upper()} API Key")
        else:
            print(f"警告: 未找到 {args.api_provider.upper()} API Key，请通过以下方式之一设置:")
            if args.api_provider == 'zhipuai':
                print("  1. 在config.py中设置 ZHIPUAI_API_KEY")
                print("  2. 设置环境变量 ZHIPUAI_API_KEY")
            else:
                print("  1. 在config.py中设置 OPENAI_API_KEY")
                print("  2. 设置环境变量 OPENAI_API_KEY")
            print("  3. 使用命令行参数 --api_key")

    return args


def main():
    print("=" * 60)
    print("Self-Refinement Runner - 自我修正便捷运行脚本")
    print("=" * 60)

    args = parse_args()

    print("\n配置信息:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  分割: {args.split}")
    print(f"  最大轮数: {args.maximum_rounds}")
    print(f"  备份策略: {args.backup_strategy}")
    print(f"  LLM 备份结果路径: {args.backup_LLM_result_path}")
    print(f"  API 提供商: {args.api_provider}")
    print(f"  自我修正模型: {args.model_name}")
    print(f"  最大新 token 数: {args.max_new_tokens}")
    print(f"  停止词: {args.stop_words}")
    print()

    try:
        for round_idx in range(1, args.maximum_rounds + 1):
            print("=" * 60)
            print(f"开始第 {round_idx} 轮自我修正...")
            print("=" * 60)

            engine = SelfRefinementEngine(args, round_idx)
            engine.single_round_self_refinement()

        print()
        print("=" * 60)
        print("自我修正全部轮次完成!")
        print("=" * 60)
        print(f"结果已保存到: ./outputs/logic_programs/self-refine-{{1..{args.maximum_rounds}}}_{args.dataset_name}_{args.split}_{args.model_name}.json")

    except KeyboardInterrupt:
        print("\n\n用户中断了程序执行")
        sys.exit(1)
    except Exception as exc:
        print(f"\n错误: 自我修正过程中出现异常 - {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


