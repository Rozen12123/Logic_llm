#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
便捷运行脚本 - 用于运行 models/logic_program.py
使用 vLLM 后端（通过 Nginx 负载均衡器）
"""

# ============================================================================
# 快速配置区域 - 在这里修改常用的运行参数
# ============================================================================

# 数据集配置
DATASET_NAME = 'FOLIO'  # 可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'
DATASET_SPLIT = 'dev'      # 可选: 'dev', 'test', 'train'

# vLLM 配置（通过 Nginx）
VLLM_API_URL = "http://localhost:8668/v1"  # Nginx 负载均衡器地址
VLLM_API_KEY = "sk-12345678"               # vLLM 启动时设置的固定密钥
MODEL_NAME = "Qwen3-8B-1203"                  # 模型名称（根据实际部署的模型修改）

# 其他配置（一般不需要修改）
DATA_PATH = './data'
SAVE_PATH = './output_data/programs/logic_programs_Qwen3-8B-1203-think'
MAX_NEW_TOKENS = 10000  # 最大新token数
STOP_WORDS = '------'
BATCH_SIZE = 50  # 每个批次处理的样本数，建议设置为并发数的 1-2 倍
MAX_CONCURRENT = 50  # 最大并发请求数
MAX_RETRIES = 1

# 提示词模式配置
USE_SYSTEM_MESSAGE = False # True: 使用 system message 模式（从 prompts_qwen_system 和 prompts_qwen_nothinking 加载）
                            # False: 使用传统模式（从 prompts 加载完整 prompt）

# ============================================================================
# 以下为代码实现部分，一般不需要修改
# ============================================================================

import os
import sys
import argparse
from config_loader import load_config

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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='便捷运行脚本 - 用于运行 models/logic_program.py（使用 vLLM 后端）\n\n'
                    '提示: 可以直接在脚本文件开头修改配置（DATASET_NAME, MODEL_NAME等），'
                    '也可以使用命令行参数覆盖这些配置。\n\n'
                    '使用前请确保：\n'
                    '1. 已通过 start_cluster.sh 启动 vLLM 集群\n'
                    '2. 已通过 nginx_m.sh start 启动 Nginx 负载均衡器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 方式1: 直接运行（使用文件开头的默认配置）
  python run_logic_program_vllm.py

  # 方式2: 使用命令行参数覆盖配置
  python run_logic_program_vllm.py --dataset_name ProntoQA --model_name Qwen3-8B-5

  # 方式3: 只覆盖部分参数
  python run_logic_program_vllm.py --dataset_name FOLIO --split test

支持的数据集: {', '.join(SUPPORTED_DATASETS)}
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
        choices=['dev', 'test', 'train'],
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
        '--api_url',
        type=str,
        default=VLLM_API_URL,
        help=f'vLLM API URL (默认: {VLLM_API_URL})'
    )
    
    parser.add_argument(
        '--api_key',
        type=str,
        default=VLLM_API_KEY,
        help=f'vLLM API Key (默认: {VLLM_API_KEY})'
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
    
    parser.add_argument(
        '--use_system_message',
        action='store_true',
        default=USE_SYSTEM_MESSAGE,
        help=f'使用 system message 模式（从 prompts_qwen_system 和 prompts_qwen_nothinking 加载提示词）(默认: {USE_SYSTEM_MESSAGE})'
    )
    
    args = parser.parse_args()
    
    # 设置 API 提供商为 openai（因为 vLLM 使用 OpenAI 兼容接口）
    args.api_provider = 'openai'
    # 设置 base_url 为 vLLM API URL
    args.base_url = args.api_url
    
    return args


def check_vllm_connection(api_url: str, api_key: str) -> bool:
    """检查 vLLM 连接是否正常"""
    try:
        import requests
    except ImportError:
        print("⚠️  警告: requests 库未安装，跳过连接检查")
        print("   可以运行: pip install requests")
        return True  # 如果 requests 未安装，跳过检查但继续运行
    
    try:
        # 尝试发送一个简单的请求来检查连接
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(f"{api_url}/models", headers=headers, timeout=5)
        if response.status_code == 200:
            return True
        else:
            print(f"⚠️  警告: vLLM API 返回状态码 {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 错误: 无法连接到 vLLM API")
        print(f"   请确保:")
        print(f"   1. vLLM 集群已启动 (运行 start_cluster.sh)")
        print(f"   2. Nginx 负载均衡器已启动 (运行 nginx_m.sh start)")
        print(f"   3. API URL 正确: {api_url}")
        return False
    except Exception as e:
        print(f"⚠️  警告: 检查 vLLM 连接时出错: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("Logic Program Generator - vLLM 版本")
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
    print(f"  API提供商: vLLM (通过 Nginx)")
    print(f"  API URL: {args.api_url}")
    print(f"  模型: {args.model_name}")
    print(f"  最大新token数: {args.max_new_tokens}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  最大并发数: {args.max_concurrent}")
    print(f"  最大重试次数: {args.max_retries}")
    print(f"  使用 System Message: {args.use_system_message}")
    print()
    
    # 检查 vLLM 连接
    print("检查 vLLM 连接...")
    if not check_vllm_connection(args.api_url, args.api_key):
        print("\n❌ vLLM 连接失败，程序退出")
        sys.exit(1)
    print("✅ vLLM 连接正常")
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

