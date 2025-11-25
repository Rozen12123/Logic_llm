#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷运行脚本 - 用于运行 models/logic_program.py
支持选择数据集、模型和API提供商
"""

# ============================================================================
# 快速配置区域 - 在这里修改常用的运行参数
# ============================================================================

# 数据集配置
DATASET_NAME = 'LogicalDeduction'  # 可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'
DATASET_SPLIT = 'dev'      # 可选: 'dev', 'test'

# API配置
API_PROVIDER = 'zhipuai'     # 可选: 'openai', 'zhipuai', 'iflow'
MODEL_NAME = 'glm-4.5'  # iflow模型: 'TBStars2-200B-A13B'
                                   # 智谱AI模型: 'GLM-4.6', 'glm-4', 'glm-4-flash-250414', 'glm-3-turbo'
                                   # OpenAI模型: 'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'text-davinci-003'

# 其他配置（一般不需要修改）
DATA_PATH = './data'
SAVE_PATH = './outputs/logic_programs'
MAX_NEW_TOKENS = 20000  # 增加到 2048 以避免输出被截断
STOP_WORDS = '------'
BATCH_SIZE = 1
MAX_CONCURRENT = 1
MAX_RETRIES = 1

# ============================================================================
# 以下为代码实现部分，一般不需要修改
# ============================================================================

import os
import sys
import argparse
from config_loader import load_config, load_api_key, load_api_provider, load_base_url

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 logic_program 模块
try:
    from models.logic_program import LogicProgramGenerator
except ImportError as e:
    print(f"错误: 无法导入必要模块: {e}")
    print("请确保已安装所有依赖: pip install -r requirements.txt")
    sys.exit(1)

# 支持的数据集
SUPPORTED_DATASETS = ['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT']

# 支持的API提供商
SUPPORTED_PROVIDERS = ['openai', 'zhipuai', 'iflow']

# OpenAI 常用模型
OPENAI_MODELS = ['text-davinci-003', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo']

# 智谱AI 常用模型
ZHIPUAI_MODELS = ['GLM-4.6', 'glm-4', 'glm-4-flash-250414', 'glm-3-turbo']

# iflow 常用模型
IFLOW_MODELS = ['TBStars2-200B-A13B']


def get_default_config():
    """获取默认配置"""
    config = load_config()
    return {
        'api_provider': config.get('api_provider', 'zhipuai'),
        'api_key': config.get('api_key'),
        'model_name': config.get('model_name', 'GLM-4.6'),
        'max_new_tokens': config.get('max_new_tokens', 2048),  # 默认增加到 2048
        'stop_words': config.get('stop_words', '------'),
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='便捷运行脚本 - 用于运行 models/logic_program.py\n\n'
                    '提示: 可以直接在脚本文件开头修改配置（DATASET_NAME, API_PROVIDER, MODEL_NAME等），'
                    '也可以使用命令行参数覆盖这些配置。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 方式1: 直接运行（使用文件开头的默认配置）
  python run_logic_program.py

  # 方式2: 使用命令行参数覆盖配置
  python run_logic_program.py --dataset_name ProntoQA --model_name glm-4 --api_provider zhipuai

  # 方式3: 只覆盖部分参数
  python run_logic_program.py --dataset_name FOLIO --split test

  # 使用OpenAI API
  python run_logic_program.py --api_provider openai --model_name gpt-4

支持的数据集: {', '.join(SUPPORTED_DATASETS)}
支持的API提供商: {', '.join(SUPPORTED_PROVIDERS)}
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
        '--data_path',
        type=str,
        default=DATA_PATH,
        help=f'数据路径 (默认: {DATA_PATH})'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default=SAVE_PATH,
        help=f'保存路径 (默认: {SAVE_PATH})'
    )
    
    parser.add_argument(
        '--api_provider',
        type=str,
        default=API_PROVIDER,
        choices=SUPPORTED_PROVIDERS,
        help=f'API提供商 (默认: {API_PROVIDER})'
    )
    
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API Key (默认: 从config.py或环境变量读取)'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default=MODEL_NAME,
        help=f'模型名称 (默认: {MODEL_NAME})'
    )
    
    parser.add_argument(
        '--stop_words',
        type=str,
        default=STOP_WORDS,
        help=f'停止词 (默认: {STOP_WORDS})'
    )
    
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=MAX_NEW_TOKENS,
        help=f'最大新token数 (默认: {MAX_NEW_TOKENS})'
    )
    
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=MAX_CONCURRENT,
        help=f'最大并发数 (默认: {MAX_CONCURRENT})'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'批处理大小 (默认: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--max_retries',
        type=int,
        default=MAX_RETRIES,
        help=f'最大重试次数 (默认: {MAX_RETRIES})'
    )
    
    args = parser.parse_args()
    
    # 如果命令行没有提供api_key，尝试从配置文件或环境变量读取
    if not args.api_key:
        args.api_key = load_api_key(args.api_provider)
        if args.api_key:
            print(f"已从配置文件或环境变量读取{args.api_provider.upper()} API Key")
        else:
            print(f"\n警告: 未找到{args.api_provider.upper()} API Key!")
            print(f"请通过以下方式之一设置:")
            if args.api_provider == 'zhipuai':
                print("  1. 在config.py中设置ZHIPUAI_API_KEY")
                print("  2. 设置环境变量ZHIPUAI_API_KEY")
            elif args.api_provider == 'iflow':
                print("  1. 在config.py中设置IFLOW_API_KEY")
                print("  2. 设置环境变量IFLOW_API_KEY")
            else:
                print("  1. 在config.py中设置OPENAI_API_KEY")
                print("  2. 设置环境变量OPENAI_API_KEY")
            print("  3. 使用命令行参数 --api_key")
            print("\n程序将退出，请先配置API Key。")
            sys.exit(1)
    
    # 加载 base_url（用于 iflow 等需要自定义 base_url 的 API）
    args.base_url = load_base_url(args.api_provider)
    
    return args


def main():
    """主函数"""
    print("=" * 60)
    print("Logic Program Generator - 便捷运行脚本")
    print("=" * 60)
    print()
    
    # 解析参数
    args = parse_args()
    
    # 显示配置信息
    print("\n配置信息:")
    print(f"  数据集: {args.dataset_name}")
    print(f"  分割: {args.split}")
    print(f"  数据路径: {args.data_path}")
    print(f"  保存路径: {args.save_path}")
    print(f"  API提供商: {args.api_provider}")
    print(f"  模型: {args.model_name}")
    print(f"  最大新token数: {args.max_new_tokens}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  最大并发数: {args.max_concurrent}")
    print(f"  最大重试次数: {args.max_retries}")
    print()
    
    # 检查数据路径是否存在
    dataset_path = os.path.join(args.data_path, args.dataset_name)
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 检查数据集文件是否存在
    dataset_file = os.path.join(dataset_path, f'{args.split}.json')
    if not os.path.exists(dataset_file):
        print(f"错误: 数据集文件不存在: {dataset_file}")
        sys.exit(1)
    
    # 创建保存目录
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print(f"已创建保存目录: {args.save_path}")
    
    print("=" * 60)
    print("开始生成逻辑程序...")
    print("=" * 60)
    print()
    
    try:
        # 创建生成器
        logic_program_generator = LogicProgramGenerator(args)
        
        # 运行批处理生成
        logic_program_generator.batch_logic_program_generation(
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        print()
        print("=" * 60)
        print("生成完成!")
        print("=" * 60)
        output_file = os.path.join(
            args.save_path,
            f'{args.dataset_name}_{args.split}_{args.model_name}.json'
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

