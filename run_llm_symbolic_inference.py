#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接使用大模型对符号逻辑程序进行推理，生成与 outputs/logic_inference 相同格式的结果文件。
"""

# ============================================================================
# 快速配置区域 - 可按需修改默认值，也可通过命令行参数覆盖
# ============================================================================

# 默认数据集（可选: 'ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT'）
DATASET_NAME = 'AR-LSAT'
DATASET_SPLIT = 'dev'  # 可选: 'dev', 'test'，'train'

API_PROVIDER = 'iflow'      # 可选: 'openai', 'zhipuai', 'iflow'
MODEL_NAME = 'glm-4.6'
STOP_WORDS = '------'
MAX_NEW_TOKENS = 20000
TEMPERATURE = 0.0
MAX_CONCURRENT = 1           # 控制并发 API 调用
BATCH_SIZE = 1               # 每次批量请求的样本数

DEFAULT_LOGIC_PROGRAM_FILE = './outputs/logic_programs/AR-LSAT_dev_glm-4.6.json'
SAVE_PATH = './output_data/inference/logic_inference_qwen3-8b_llm_all'
OUTPUT_SUFFIX = '_llm-symbolic'   # 结果文件会追加该后缀

MAX_EXAMPLES = None           # 仅用于调试，限制每个文件处理的样本数量
DRY_RUN = False               # True 时只打印 prompt 示例，不实际调用 API

# ============================================================================
# 代码实现
# ============================================================================

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 确保可以导入 models/utils.py
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

try:
    from models.utils import OpenAIModel, ZhipuAIModel
except ImportError:
    from models.utils import OpenAIModel, ZhipuAIModel  # 兜底

from config_loader import load_api_key, load_api_provider, load_base_url

SUPPORTED_DATASETS = ['ProntoQA', 'ProofWriter', 'FOLIO', 'LogicalDeduction', 'AR-LSAT']
SUPPORTED_PROVIDERS = ['openai', 'zhipuai', 'iflow']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='让大模型直接阅读 outputs/logic_programs 中的符号程序，并生成 inference 结果文件',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset_name', type=str, default=DATASET_NAME, choices=SUPPORTED_DATASETS)
    parser.add_argument('--split', type=str, default=DATASET_SPLIT, choices=['dev', 'test'])
    parser.add_argument('--logic_program_file', type=str, default=DEFAULT_LOGIC_PROGRAM_FILE,
                        help='要读取的逻辑程序文件路径（默认直接使用 DEFAULT_LOGIC_PROGRAM_FILE）')
    parser.add_argument('--save_path', type=str, default=SAVE_PATH,
                        help='推理结果输出目录')
    parser.add_argument('--output_suffix', type=str, default=OUTPUT_SUFFIX,
                        help='输出文件名追加的后缀')
    parser.add_argument('--api_provider', type=str, default=API_PROVIDER, choices=SUPPORTED_PROVIDERS)
    parser.add_argument('--model_name', type=str, default=MODEL_NAME)
    parser.add_argument('--api_key', type=str, default=None,
                        help='API Key（未提供时自动从配置或环境变量读取）')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI 兼容接口的 base_url（如 iflow）')
    parser.add_argument('--stop_words', type=str, default=STOP_WORDS)
    parser.add_argument('--max_new_tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--max_concurrent', type=int, default=MAX_CONCURRENT,
                        help='并发请求数，<=1 表示串行')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='每次批量调用的样本数')
    parser.add_argument('--max_examples', type=int, default=MAX_EXAMPLES,
                        help='每个逻辑程序文件最多处理的样本数（调试用）')
    parser.add_argument('--dry_run', action='store_true', default=DRY_RUN,
                        help='仅打印 prompt 示例，不实际调用 API')
    parser.add_argument('--show_prompt_sample', action='store_true',
                        help='打印第一条 prompt，便于调试')
    parser.add_argument('--skip_existing', action='store_true',
                        help='若输出文件已存在则跳过处理')
    return parser.parse_args()


def ensure_api_credentials(args: argparse.Namespace) -> argparse.Namespace:
    if not args.api_provider:
        args.api_provider = load_api_provider()
    if not args.api_key:
        args.api_key = load_api_key(args.api_provider)
    if not args.api_key:
        raise ValueError(f"未找到 {args.api_provider} 的 API Key，请通过配置文件或环境变量提供。")
    if args.api_provider != 'zhipuai' and not args.base_url:
        args.base_url = load_base_url(args.api_provider)
    return args


def create_model_client(args: argparse.Namespace):
    if args.api_provider == 'zhipuai':
        return ZhipuAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
    else:
        return OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, base_url=args.base_url)


def collect_logic_program_files(args: argparse.Namespace) -> List[str]:
    """
    只处理单个逻辑程序文件。若需要切换文件，请修改 DEFAULT_LOGIC_PROGRAM_FILE 或通过命令行参数覆盖。
    """
    if not args.logic_program_file:
        raise ValueError("必须提供 logic_program_file，可通过 DEFAULT_LOGIC_PROGRAM_FILE 或命令行参数设置。")
    target = os.path.abspath(args.logic_program_file)
    if not os.path.exists(target):
        raise FileNotFoundError(f"指定的逻辑程序文件不存在: {target}")
    return [target]


def get_logic_program_text(example: Dict[str, Any]) -> str:
    programs = example.get('raw_logic_programs') or []
    if isinstance(programs, str):
        return programs.strip()
    if isinstance(programs, list):
        cleaned = [p.strip() for p in programs if isinstance(p, str) and p.strip()]
        return '\n\n'.join(cleaned)
    return ''


def format_options(options: Any) -> str:
    if isinstance(options, list):
        lines = []
        for item in options:
            if isinstance(item, str):
                lines.append(item.strip())
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"({item[0]}) {item[1]}")
            elif isinstance(item, dict):
                label = item.get('label') or item.get('option') or item.get('id')
                text = item.get('text') or item.get('value') or item.get('statement')
                if label and text:
                    lines.append(f"({label}) {text}")
                else:
                    lines.append(json.dumps(item, ensure_ascii=False))
            else:
                lines.append(str(item))
        return '\n'.join(lines) if lines else 'N/A'
    if isinstance(options, dict):
        return '\n'.join(f"{k}: {v}" for k, v in options.items())
    return str(options) if options else 'N/A'


PROMPT_TEMPLATE = """You are a meticulous symbolic reasoning assistant. Rely only on the provided problem statement and symbolic logic program to determine the single correct option.

### Problem Statement
Context:
{context}

Question:
{question}

Options:
{options_text}

### Symbolic Logic Program
{logic_program}

Your reply must be a JSON string in the following format:
{{
  "answer": "One letter from A/B/C/D/E",
  "rationale": "Short explanation grounding the choice in the logic program"
}}
"""


def build_prompt(example: Dict[str, Any]) -> str:
    context = example.get('context', '').strip()
    question = example.get('question', '').strip()
    options_text = format_options(example.get('options'))
    logic_program = get_logic_program_text(example)
    if not logic_program:
        logic_program = "（未提供符号逻辑程序，若无法判断请返回错误信息。）"
    return PROMPT_TEMPLATE.format(
        context=context or "(missing context)",
        question=question or "(missing question)",
        options_text=options_text,
        logic_program=logic_program
    ).strip()


JSON_PATTERN = re.compile(r'\{.*\}', re.DOTALL)


def extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    match = JSON_PATTERN.search(text)
    if match:
        snippet = match.group(0)
        try:
            json.loads(snippet)
            return snippet
        except Exception:
            return None
    return None


CHOICE_PATTERN = re.compile(r'\b([A-E])\b', re.IGNORECASE)


def normalize_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = text.strip().upper()
    if len(text) == 1 and text in 'ABCDE':
        return text
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return match.group(1)
    match = re.search(r'\(([A-E])\)', text)
    if match:
        return match.group(1)
    return None


def parse_llm_response(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    返回 (choice, rationale, error_message)
    """
    if not raw_text or not raw_text.strip():
        return None, None, "空响应"
    json_block = extract_json_block(raw_text)
    if json_block:
        try:
            data = json.loads(json_block)
            answer = data.get('answer') or data.get('choice') or data.get('prediction')
            rationale = data.get('rationale') or data.get('explanation') or data.get('reasoning')
            choice = normalize_choice(answer if isinstance(answer, str) else str(answer))
            if choice:
                return choice, rationale, None
            return None, rationale, f"无法从 JSON 中解析选项: {answer}"
        except json.JSONDecodeError as exc:
            return None, None, f"JSON 解析失败: {exc}"
    choice = normalize_choice(raw_text)
    if choice:
        return choice, None, None
    return None, None, "未能解析出选项"


def run_inference_on_file(
    file_path: str,
    api_client,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        raise ValueError(f"逻辑程序文件格式错误（应为列表）: {file_path}")

    if args.max_examples:
        examples = examples[:args.max_examples]

    prompts = [build_prompt(example) for example in examples]

    if args.show_prompt_sample and prompts:
        print("\n==== Prompt 示例 ====")
        print(prompts[0])
        print("=====================\n")

    if args.dry_run:
        print("dry_run 模式：未调用 API，直接退出。")
        return {
            'file_path': file_path,
            'results': [],
        }

    total_examples = len(examples)
    if total_examples == 0:
        return {
            'file_path': file_path,
            'results': []
        }

    predictions: List[Dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    progress_desc = f"推理中: {os.path.basename(file_path)}"
    progress_bar = tqdm(total=total_examples, desc=progress_desc, unit="sample")

    try:
        for start_idx in range(0, total_examples, batch_size):
            end_idx = min(start_idx + batch_size, total_examples)
            chunk_examples = examples[start_idx:end_idx]
            chunk_prompts = prompts[start_idx:end_idx]
            try:
                responses = api_client.batch_generate(
                    chunk_prompts,
                    temperature=args.temperature,
                    max_concurrent=args.max_concurrent
                )
            except Exception as exc:
                responses = [None] * len(chunk_prompts)
                print(f"调用 API 失败: {exc}")
            if not isinstance(responses, list):
                responses = [responses]
            if len(responses) != len(chunk_examples):
                responses = (responses + [None] * len(chunk_examples))[:len(chunk_examples)]
            for example, raw in zip(chunk_examples, responses):
                raw_text = raw if isinstance(raw, str) else (json.dumps(raw, ensure_ascii=False) if raw else '')
                choice, rationale, error = parse_llm_response(raw_text)
                result_entry = {
                    'id': example.get('id'),
                    'context': example.get('context'),
                    'question': example.get('question'),
                    'answer': example.get('answer'),
                    'options': example.get('options'),
                    'predicted_answer': choice,
                    'rationale': rationale,
                    'flag': 'success' if choice else 'failed',
                    'error_message': error,
                    'raw_response': raw_text,
                    'source_logic_program_file': os.path.basename(file_path)
                }
                predictions.append(result_entry)
            progress_bar.update(len(chunk_examples))
    finally:
        progress_bar.close()

    return {
        'file_path': file_path,
        'results': predictions,
    }


def write_results(file_path: str, results: List[Dict[str, Any]], output_dir: str, suffix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(file_path)
    stem = basename[:-5] if basename.endswith('.json') else basename
    output_name = f"{stem}{suffix}.json"
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return output_path


def should_skip_file(output_dir: str, file_path: str, suffix: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    stem = os.path.basename(file_path)
    stem = stem[:-5] if stem.endswith('.json') else stem
    output_path = os.path.join(output_dir, f"{stem}{suffix}.json")
    return os.path.exists(output_path)


def main():
    args = parse_args()
    args = ensure_api_credentials(args)

    print("=" * 60)
    print("LLM Symbolic Inference Runner")
    print("=" * 60)
    print(f"数据集: {args.dataset_name} ({args.split})")
    print(f"模型: {args.api_provider} / {args.model_name}")
    print(f"逻辑程序文件: {args.logic_program_file}")
    print(f"输出目录: {args.save_path}")
    if args.max_examples:
        print(f"每个文件最多处理 {args.max_examples} 条样本")
    if args.dry_run:
        print("当前为 dry_run 模式，不会调用 API。")
    print("=" * 60)

    logic_files = collect_logic_program_files(args)

    api_client = create_model_client(args)

    for idx, file_path in enumerate(logic_files, start=1):
        if should_skip_file(args.save_path, file_path, args.output_suffix, args.skip_existing):
            print(f"[{idx}/{len(logic_files)}] 跳过（已存在结果）: {file_path}")
            continue

        print(f"[{idx}/{len(logic_files)}] 处理: {file_path}")
        try:
            outcome = run_inference_on_file(file_path, api_client, args)
            results = outcome.get('results', [])
            output_path = write_results(file_path, results, args.save_path, args.output_suffix)
            print(f"    完成，写入: {output_path} (共 {len(results)} 条)")
        except Exception as exc:
            print(f"    处理失败: {exc}")


if __name__ == '__main__':
    main()

